"""
  Repiar the extracted samples from imojie based on the paraphrased data.
"""
import re
from multiprocessing import cpu_count, Process, Lock, Manager
import math, time
import numpy as np
from tqdm import tqdm
from stanza.server import CoreNLPClient
import argparse
import os
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, T5Model
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor 

STANFORD_CORENLP = 'AESOP/evaluation/apps/stanford-corenlp-full-2018-10-05'

os.environ["CORENLP_HOME"] = STANFORD_CORENLP
os.environ["_JAVA_OPTIONS"] = '-Djava.io.tmpdir=/tmp'

def legal_check(argument):
    if len(argument)<=1:
        return False
    words = argument.strip().split()
    matches = []
    for w in words:
        if re.match(r'.*?(\w).*$', w):
            matches.append(True)
        else:
            matches.append(False)
    if not any(matches):
        return False
    return True


def repair_arg_a_sample(sample, tree):
    """ Repair a sample of imojie input format.
      Return: A repaired sample if the original sample can be repaired, otherwise a "" string.
    """
    pattern = re.compile(
            r'(.*?)\t<arg1> (.*?) </arg1> <rel> (.*?) </rel> <arg2> (.*?) </arg2>\t(.*?)$')
    items = pattern.match(sample)
    repaired_sample = sample.rstrip()
    if items:
        sent, arg1, rel, arg2, score = items.groups()
        # basic legal check
        if not (legal_check(arg1) or legal_check(arg2)):
            return ""
        # argument omission repair
        new_arg1_arg2 = []
        new_arg1_arg2_indices = []
        sent_tokens = sent.split()
        for arg in (arg1, arg2):
            arg_tokens = arg.split()
            head = get_head_index(sent_tokens, arg_tokens, return_all=False)
            if head is None:
                # print(sample)
                return sample.rstrip()
            tail = head + len(arg_tokens)
            # complete from syntactic tree
            arg_head, arg_tail = build_entity_pivot(
                        arg, sent_tokens, head, tail, tree)
            new_arg1_arg2.append(sent_tokens[arg_head : arg_tail])
            new_arg1_arg2_indices.append([arg_head, arg_tail])
        
        # the maximum length of argument threshold
        [[a1,a2], [b1,b2]] = new_arg1_arg2_indices
        if (a2-a1) < round(1.2*len(arg1.split())) and (b2-b1) < round(1.2*len(arg2.split())) \
            and ((a1<b1 and a2<b2) or (a1>b1 and a2>b2)):
            repaired_sample = "{}\t<arg1> {} </arg1> <rel> {} </rel> <arg2> {} </arg2>\t{}".format(sent, " ".join(new_arg1_arg2[0]), rel, " ".join(new_arg1_arg2[1]), score)
        
    return repaired_sample


def repair_arg_batch_samples(samples, idx, sync_dict, lock, cache_file=None):
    """ Repair a batch of samples in a specific cpu.
    """
    stanford_client = CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'parse'],
            endpoint=f"http://localhost:{9000+int(idx)}",
            timeout=30000,
            memory='6G',
            be_quiet=True,
            properties={
                "ssplit.eolonly": True}
        )
    
    # repair
    rts = []
    for sample in samples:
        # sent = re.match(r'(.*?)\t<arg1>.*', sample).group(1)
        match = re.match(r'(.*?)\t<arg1>.*', sample)
        if match:
            sent = match.group(1)
            ann = stanford_client.annotate(sent)
            if len(ann.sentence) > 0:
                re_sample = repair_arg_a_sample(sample, ann.sentence[0].parseTree)
                if re_sample is not "": # legal
                    rts.append(re_sample)
    
    lock.acquire()
    sync_dict[str(idx)] = rts
    lock.release()
    
    # clean
    stanford_client.stop()
    if stanford_client.server_props_path:
        if os.path.isfile(stanford_client.server_props_path):
            os.remove(stanford_client.server_props_path)


def repair_argments(in_file, out_file, n_cpus, cache_dir=None, carb_format=False):
    """ Repair a file of imojie input format.
    """ 
    with open(in_file, "r") as f:
        file_size = sum([1 for i,_ in enumerate(f)])
    print(f"Total number of sentences: {file_size}")

    lock = Lock()
    cpus = n_cpus if n_cpus != None else cpu_count()
    sync_dict = Manager().dict()

    with open(in_file, "r") as f:
        # segment
        piece = math.ceil(file_size / float(cpus))
        # run multi-process
        process_list = []
        for i in range(min(cpus, math.ceil(file_size/piece))):
            lines = list(itertools.islice(f,int(piece)))
            cache = os.path.join(cache_dir, f"{in_f_name}.cache{i}") if cache_dir is not None else None
            p = Process(target=repair_arg_batch_samples, args=(lines, i, sync_dict ,lock, cache))
            process_list.append(p)
        start = time.time()
        for p in process_list:
            p.start()
        for p in process_list:
            p.join()
        end = time.time()
        print('Total time consuming:[%.4f]' % (end - start))
    
    # arrange to the original order
    rts = []
    for i in range(len(sync_dict)):
        rts.extend(sync_dict[str(i)])
    with open(out_file, "w") as f:
        for line in rts:
            if line is not "":
                if not carb_format:
                    f.write(line + "\n")
                else:
                    pattern = re.compile(
            r'(.*?)\t<arg1> (.*?) </arg1> <rel> (.*?) </rel> <arg2> (.*?) </arg2>\t(.*?)$')
                    items = pattern.match(sample)
                    sent, arg1, rel, arg2, score = items.groups()
                    f.write("{}\t{}\t{}\t{}".format(sent,rel,arg1,arg2))



def same_token_rank(tokens, target_idx):
    target = tokens[target_idx]
    cnt = 0
    for i in range(target_idx+1):
        if tokens[i] == target:
            cnt += 1
    return cnt


# entity token is subset sentence_tokens
def get_head_index(sentence_tokens, entity_tokens, return_all=False):
    pos = [i for i in range(len(sentence_tokens))
           if sentence_tokens[i] == entity_tokens[0]]
    if len(entity_tokens) == 1:
        if len(pos) > 0:
            if return_all:
                return pos
            else:
                return pos[0]
    else:
        ans = []
        for i in pos:
            flag = True
            if i + len(entity_tokens) > len(sentence_tokens):
                flag = False
                break
            for j in range(min(len(entity_tokens), 3)):
                if sentence_tokens[i+j] != entity_tokens[j]:
                    flag = False
                    break
            if flag:
                if return_all:
                    ans.append(i)
                else:
                    return i
        if return_all:
            return ans


def _subtree_check(tree, token, cnt, ori_name):
    stack = [(tree, 0)]
    visited = {0: 0}
    id_cnt = 1
    find_cnt = 0
    # locate the token node in parseTree
    while True:
        if len(stack) == 0:
            break
        cur, cur_id = stack[-1]
        # subtree
        if visited[cur_id] < len(cur.child):
            # add child
            stack.append((cur.child[visited[cur_id]], id_cnt))
            visited[cur_id] += 1
            visited[id_cnt] = 0
            id_cnt += 1
        else:
            stack.pop()
            if len(cur.child) == 0 and cur.value == token:
                # leave
                find_cnt += 1
                if find_cnt == cnt:
                    break
    subtree = None
    # reverse-transpose, from gradepa
    for i in stack[-2::-1]:
        if i[0].value in ['NP', 'QP', 'NX', 'WHNP']:
            subtree = i[0]
        # elif "of" in ori_name and i[0].value == "PP":
        #     subtree = i[0]
        #     break
        # elif "and" in ori_name and i[0].value == "@NP":
        #     subtree = i[0]
        #     break
        else:
            break

    ans = []
    if subtree is not None:
        stack = [(subtree, 0)]
        visited = set()
        id_cnt = 0
        while True:
            if len(stack) == 0:
                break
            cur, cur_id = stack.pop()
            if len(cur.child) > 0:
                # subtree
                if cur_id not in visited:
                    for i in cur.child[::-1]:
                        # add child
                        id_cnt += 1
                        stack.append((i, id_cnt))
            else:
                ans.append(cur.value)
            visited.add(cur_id)
    return ans


def build_entity_pivot(ori_name, gen_tokens, head, tail, tree):
    ent_head = head
    ent_tail = tail
    for rank, token in enumerate(gen_tokens[head:tail]):
        new_tokens = _subtree_check(
            tree, token, same_token_rank(gen_tokens, head+rank), ori_name)

        if len(new_tokens) > 0:
            new_head = get_head_index(gen_tokens, new_tokens)
            if new_head is not None:
                ent_head = min(new_head, ent_head)
                ent_tail = max(new_head+len(new_tokens), ent_tail)

    return ent_head, ent_tail



def train_extract_predicate(
    tokenizer,
          model,
          trainset_loader,
          devset_loader,
          save_dir,
          train_batch_size = 128,
          eval_batch_size = 56,
          dev_train_ratio = 0.01,
          learning_rate = 1e-3,
          weight_decay = 0.0001,
          grad_accum_steps = 1,
          grad_clip_max = -1,
          train_epochs = 100,
          warmup_ratio = 0.006,
          eval_steps = -1,
          shuffle=True,
          log_steps = 200,
          **kw_args):
    """ Build a model of GenerateEntities and train it.
      Args:
        @train_tsv: the training file where each line consists of a source sentence \t N-tuple.
    """
    
    seed = 71
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    print("# of train batches: %d " % len(trainset_loader))
    
    # transfer to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
        
    # build optimizer & scheduler
    total_steps = int(len(trainset_loader)*train_epochs // train_batch_size)
#     warmup_steps = int(total_steps * warmup_ratio)
# #     pretrained_param_ids = list(map(id, model.base_model.parameters()))
#     params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = optim.AdamW([
#         {"params": params, "lr": learning_rate, "weight_decay": 0.0},
#     ], lr=learning_rate)
#     warmup_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    
    # train
    best_loss = 1e10
    num_steps = 0
    for epoch in range(train_epochs):
        print("New epoch ...")
        model.zero_grad()
        with tqdm(total=len(trainset_loader), desc="Training") as t:
            for step, batch in enumerate(trainset_loader):
                num_steps += 1
                model.train()
                inputs = {"input_ids": batch[0].to(device),
#                           "attention_mask": batch[1].to(device),
                          "labels": batch[2].to(device),
#                           "decoder_attention_mask": batch[3].to(device)
                         }
                out = model(**inputs)
                logits = out.logits
                loss = out.loss

                optimizer.zero_grad()
                loss.backward()

                if num_steps % grad_accum_steps == 0:
                    if grad_clip_max > 0:
                        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_max)
                    optimizer.step()
                    model.zero_grad()
                    
                    scheduler.step()
#                     if num_steps > warmup_steps:
#                         scheduler.step()
#                 if num_steps <= warmup_steps:
#                     warmup_scheduler.step()

#                 if num_steps % log_steps == 0:
#                     print(f"Train epoch: {epoch}, step: {step}, loss: {loss}")
                t.set_description("Epoch {}:".format(epoch))
                t.set_postfix(loss="{}".format(loss.item()))
                t.update(1)

                # evaluate
                if (step+1==len(trainset_loader)-1) or (eval_steps>0 and num_steps%eval_steps==0 and step%grad_accum_steps==0):
                    model.eval()
                    eval_losses = []
                    for eval_step, eval_batch in enumerate(devset_loader):
                        eval_inputs = {
                            "input_ids": eval_batch[0].to(device),
#                             "attention_mask": eval_batch[1].to(device),
                            "labels": eval_batch[2].to(device),
#                             "decoder_attention_mask": eval_batch[3].to(device)
                         }
                        dev_out = model(**eval_inputs)
                        dev_loss = dev_out.loss
                        dev_logits = dev_out.logits
                        eval_losses.append(dev_loss.detach().item())
                    loss_mean = np.mean(eval_losses)
                    print(f"Evaluation average loss: {loss_mean}")
                    if loss_mean <= best_loss:
                        best_model_path = os.path.join(save_dir, "ckpts/best.pt")
                        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(save_dir, "ckpts/best.pt"))
                    
    return best_model_path

class PreDataset(Dataset):
    
    def __init__(self, lines, tokenizer, line_type="carb"):
        # build data
        self.line_type = line_type
        self.tokenizer = tokenizer
        
        self.features = []
        n_lines = len(lines)
        for i in tqdm(range(n_lines), desc=f"Processing {line_type}"):
            line = lines[i]
            if self.line_type == "carb":
                splits = line.strip().split("\t")
                sent, args = splits[0].strip(), splits[1:]
            else:
                pattern = re.compile(r'(.*?)\t<arg1> (.*?) </arg1> <rel> (.*?) </rel> <arg2> (.*?) </arg2>\t(.*?)$')
                items = pattern.match(line.strip())
                if items:
                    sent, arg1, rel, arg2, score = items.groups()
                    args = [rel, arg1, arg2]
                else:
                    continue
            if len(args) >= 3:
                src = "{} , {} , {}".format(sent, args[1].strip(), args[2].strip())
#                 src = "{} , {} <extra_id_0> {}".format(sent, args[1].strip(), args[2].strip())
            elif len(args) >= 2:
                src = "{} , {}".format(sent, args[1].strip())
            else:
                continue
            tgt = args[0].strip()
            
#             src_toks = self.tokenizer(src) # ori_sent + </s>
#             src_ids = src_toks["input_ids"]
#             tgt_ids = self.tokenizer.encode(tgt)

            src_ids = self.tokenizer.encode(src + "</s>")
            tgt_ids = self.tokenizer.encode(tgt + "</s>")
            feat = {
                "src_ids": src_ids,
                "tgt_ids": tgt_ids
            }
            self.features.append(feat)
        
    def __len__(self):
        return len(self.features)
    
    
    def __getitem__(self, index):
        feat = self.features[index]
        return feat

    @classmethod
    def collate_fn(self, batch):
        max_src_len = max([len(f["src_ids"]) for f in batch])
        max_tgt_len = max([len(f["tgt_ids"]) for f in batch])
        src_ids = torch.tensor([f["src_ids"]+[0]*(max_src_len-len(f["src_ids"])) for f in batch], dtype=torch.long)
        src_mask = torch.tensor([[1]*(len(f["src_ids"]))+[0]*(max_src_len-len(f["src_ids"])) for f in batch], dtype=torch.long)
        tgt_ids = torch.tensor([f["tgt_ids"]+[0]*(max_tgt_len-len(f["tgt_ids"])) for f in batch], dtype=torch.long)
        tgt_mask = torch.tensor([[1]*(len(f["tgt_ids"]))+[0]*(max_tgt_len-len(f["tgt_ids"])) for f in batch], dtype=torch.long)
        
        return src_ids, src_mask, tgt_ids, tgt_mask
        
    
    
    
def repair_predicates(
    in_tsv,
    out_tsv,
    train_pre_files,
    best_model_path=None, # whether to train
):
    # model
#     config = AutoConfig.from_pretrained("/ldata/qiji/MODELS/t5-base")
#     tokenizer = AutoTokenizer.from_pretrained("/ldata/qiji/MODELS/t5-base")
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         "/ldata/qiji/MODELS/t5-base",
#         config=config
#     )
#     model = AutoModel.from_pretrained(
#         "/ldata/qiji/MODELS/t5-base",
#         config=config
#     )

#     model = T5Model.from_pretrained(
#         "/ldata/qiji/MODELS/t5-base",
#         config=config
#     )
    
    model = T5ForConditionalGeneration.from_pretrained(
        "/ldata/qiji/MODELS/t5-base"
    )
    tokenizer = T5Tokenizer.from_pretrained("/ldata/qiji/MODELS/t5-base")
    
    
    # Train a model
    if best_model_path is None: #train
        # data
        lines = []
        for f in train_pre_files:
            with open(f, "r") as _f:
                lines.extend(_f.readlines())
        train_dataset = PreDataset(lines[: int(len(lines)*0.9):], tokenizer, line_type="imojie")
        dev_dataset =  PreDataset(lines[int(len(lines)*0.9):], tokenizer, line_type="imojie")
        trainset_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, collate_fn=PreDataset.collate_fn)
        devset_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, collate_fn=PreDataset.collate_fn)
        
        best_model_path = train_extract_predicate(tokenizer, model,trainset_loader,devset_loader, "save_models", train_epochs=20)
    else:
        model.load_state_dict(torch.load(best_model_path))
        
    # transfer to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    # Repair
    in_stream = open(in_tsv, "r")
    out_stream = open(out_tsv, "w")
    with open(in_tsv, "r") as f: n_lines=sum([1 for _ in f])
    pattern = re.compile(r'(.*?)\t<arg1> (.*?) </arg1> <rel> (.*?) </rel> <arg2> (.*?) </arg2>\t(.*?)$')
    for i in tqdm(range(n_lines), desc="Repairing predicate"):
        line = next(in_stream)
        result = pattern.match(line)
        if result:
            sent, arg1, rel, arg2, score = result.groups()
            sent = sent.strip()
            src = "{} , {} , {}".format(sent, arg1.strip(), arg1.strip())
            src_toks = tokenizer(src, return_tensors="pt") # ori_sent + </s>
            src_ids = src_toks["input_ids"]
            
            out = model.generate(
                src_ids.to(device),
                num_beams=5,
                max_length=20,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
            pre_toks = tokenizer.convert_ids_to_tokens(out[0])
            pre = tokenizer.convert_tokens_to_string(pre_toks)
            pre = pre.strip().replace("<pad>","").replace("</s>","")
            
            out_stream.write("{}\t<arg1> {} </arg1> <rel> {} </rel> <arg2> {} </arg2>\t{}\n".format(sent, arg1, pre, arg2, score))
        
    in_stream.close()
    out_stream.close()
    
    
    
def resort(in_tsv):
    resorted_rts = {}
    pattern = re.compile(r'(.*?)\t<arg1> (.*?) </arg1> <rel> (.*?) </rel> <arg2> (.*?) </arg2>\t(.*?)$')
    with open(in_tsv, "r") as f:
        for line in f:
            items = pattern.match(line.strip())
            sent, arg1, rel, arg2, score = items.groups()
            if sent not in resorted_rts:
                resorted_rts = [line]
            else:
                resorted_rts.append(line)
    # save
    with open(in_tsv, "w") as f:
        for k in resorted_rts:
            f.writelines(resorted_rts[k])

    

if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="data/para/train/paranmt-h2/extractions_from_imojie.tsv")
    parser.add_argument("--out_file", type=str, default="data/para/train/paranmt-h2/extractions_from_imojie_repair.tsv")
    parser.add_argument("--carb_format", action="store_true")
    parser.add_argument("--n_cpus", type=int, default=40)
    parsre.add_argument("--train_pre_files", nargs="+", type=str, default=["data/train/4cr_qpbo_extractions.tsv"])
    # parsre.add_argument("--train_pre_files", nargs="+", type=str, default=["data/dev/carb/extractions.tsv",
    #              "data/test/carb/extractions.tsv"])
    parser.add_argument("--best_model_path", type=str, default="save_models/ckpts/best.pt")
    args = parser.parse_args()
    
    # arg_repaired_f = args.out_file+".arg_repaired"
    # if not os.path.exists(arg_repaired_f):
    #     repair_argments(args.in_file, arg_repaired_f, args.n_cpus)
    arg_repaired_f = args.in_file

    if not os.path.exists(args.out_file):
        # os.environ["CUDA_VISIBLE_DEVICES"]="7"
        repair_predicates(
            arg_repaired_f,
            args.out_file,
            train_pre_files=args.train_pre_files,
            best_model_path=args.best_model_path
        )
    
#     # clean cache
#     os.system(ls -l | awk '{if ($9 ~ /^corenlp_server-.*?props$/) print "rm "$9 | "/bin/bash"}')