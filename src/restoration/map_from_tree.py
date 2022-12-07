import argparse
import os
import re
import torch
from torch.utils.data import Dataset
from torch import cosine_similarity
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import BertForQuestionAnswering, BertTokenizer, BertModel
from torch.utils.data.dataloader import default_collate
import heapq
from stanza.server import CoreNLPClient
import contractions

STANFORD_CORENLP = 'AESOP/evaluation/apps/stanford-corenlp-full-2018-10-05'
CACHE_DIR = "cache/aesop/stanfordnlp/out/"

BERT_MODEL = "/data/MODELS/bert-base-uncased"

os.environ["CORENLP_HOME"] = STANFORD_CORENLP
os.environ["_JAVA_OPTIONS"] = '-Djava.io.tmpdir='+CACHE_DIR


def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(lambda x: x, batch))
    if batch:
        return default_collate(batch)
    else:
        return None


class SentenceFilter():
    def __init__(self, args) -> None:
        self.search_space = []
        self.args = args
        self.client = CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'parse'],
            endpoint=f"http://localhost:{65501}",
            timeout=30000,
            memory='6G',
            be_quiet=True,
            properties={
                "ssplit.eolonly": True}
        )
        self.tsv_format = args.tsv_format # openie4 or carb
        # saved files
        self.ent_result = os.path.join(self.args.out_folder, 
            self.args.para_tgt.split("/")[-1].replace("tgt", "restore.") + self.args.out_file_prefix + "only_ent.tsv")
        self.final_result = os.path.join(self.args.out_folder, 
            self.args.para_tgt.split("/")[-1].replace("tgt", "restore.") + self.args.out_file_prefix + "final.tsv")
        self.refine_mapping = os.path.join(self.args.out_folder, 
            self.args.para_tgt.split("/")[-1].replace("tgt", "restore.") + self.args.out_file_prefix + "refine_mapping.tsv")

    def __del__(self):
        if self.client.server_props_path:
            if os.path.isfile(self.client.server_props_path):
                os.remove(self.client.server_props_path)

    def init_filter(self):
        print(f"loading triples from {self.args.triples_tsv}...")
        self.source_sents = []
        self.entities = []
        temp = []

        # with open(self.args.triples_tsv) as f:
        #     line = f.readline()
        #     self.source_sents.append(line.strip())
        #     while line:
        #         line = f.readline()
        #         if line.strip():
        #             if line[:3] == '1 (':
        #                 result = line.strip()[3:-1].split(" ; ")
        #                 if len(result) < 3:
        #                     temp.append([i.strip()
        #                                 for i in result]+['']*(3-len(result)))
        #                 else:
        #                     temp.append([i.strip() for i in result])
        #         else:
        #             line = f.readline()
        #             if line:
        #                 self.source_sents.append(line.strip())
        #                 self.entities.append(temp)
        #                 temp = []
        # self.entities.append(temp)

        if self.tsv_format == "openie4":
            pattern = re.compile(r'(.*?)\t<arg1> (.*?)</arg1> <rel> (.*?)</rel> <arg2> (.*?)</arg2>')
        elif self.tsv_format == "carb":
            pattern = re.compile(r"(.*?)\t(.*?)\t(.*?)\t(.*?)[\t|\n]")
        else:
            raise BaseException("Unsupported format")
        temp = []
        with open(self.args.triples_tsv) as f:
            for line in f:
                result = pattern.match(line)
                if result:
                    if self.tsv_format == "openie4":
                        sent, ent1, rel, ent2 = result.groups()
                    else:
                        sent, rel, ent1, ent2 = result.groups()
                    if len(self.source_sents) == 0:
                        self.source_sents.append(sent)
                    elif self.source_sents[-1] != sent:
                        self.source_sents.append(sent)
                        self.entities.append(temp)
                        temp = []
                    temp.append((ent1.strip(), rel.strip(), ent2.strip()))
            if len(temp) > 0:
                self.entities.append(temp)

        assert len(self.entities) == len(self.source_sents)

        print(f"loading paraphrase from {self.args.para_source} ...")
        _hot = None
        temp = []
        self.gen_sents = []
        with open(self.args.para_source) as f_s:
            with open(self.args.para_tgt) as f_t:
                for line in f_s:
                    if not _hot:
                        _hot = line.split("<sep>")[0]
                    elif _hot != line.split("<sep>")[0]:
                        _hot = line.split("<sep>")[0]
                        self.gen_sents.append(temp)
                        temp = []
                    tgt_line = f_t.readline()
                    if tgt_line not in temp:
                        temp.append(tgt_line)
                if len(temp) > 0:
                    self.gen_sents.append(temp)

        assert len(self.entities) == len(self.gen_sents)

        print("loading tokenizer ...")
        # self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        print("ready for processing.")

    def _get_spans(self, ori_name, gen_tokens, entities_similarity, limit, tree, topk=5, not_entity=False, bert_sim_only=False):
        head = 0
        tail = head+1
        raw = []
        while True:
            # Merge consecutive spans that is matched successively as (head,tail)
            while entities_similarity[head] == 0:
                head += 1
                if head >= entities_similarity.shape[0]:
                    break
            tail = head+1
            if tail < entities_similarity.shape[0]:
                while not entities_similarity[tail] == 0:
                    tail += 1
                    if tail >= entities_similarity.shape[0]:
                        break
            if head >= entities_similarity.shape[0]:
                break

            if not_entity:
                ent_head = head
                ent_tail = tail
            else:
                if not bert_sim_only:
                    # Extend left and right bound further if it is head/tail entity
                    ent_head, ent_tail = build_entity_pivot(
                        ori_name, gen_tokens, head, tail, tree)
                else:
                    ent_head = head
                    ent_tail = tail

            if ent_head < ent_tail:
                raw.append(
                    (entities_similarity[ent_head:ent_tail].sum().item(), ent_head, ent_tail))
            head = tail
            if head >= entities_similarity.shape[0]:
                break
        ans = []
        if len(raw) > 1:
            for i in range(len(raw)):
                ans.append(raw[i])
                for j in range(i+1, len(raw)):
                    if raw[j][1]-raw[i][1] < limit and raw[j][2]-raw[i][1] < 1.2*limit:
                        ans.append((raw[j][0]+raw[i][0], raw[i][1], raw[j][2]))
            return heapq.nlargest(topk, ans, lambda x: x[0])
        else:
            return raw

    def _search_best_triple(self, depth, max_overlap=0, path=[], overlap=0):
        flag = False
        new_overlap = overlap
        for choice in self.search_space[depth]:
            for i in path:
                temp = choice[2]-choice[1]+i[2]-i[1] - \
                    (max(i[2], choice[2])-min(i[1], choice[1]))
                if temp > 0:
                    new_overlap += temp
            if depth+1 < len(self.search_space):
                ans, flag = self._search_best_triple(
                    depth+1, max_overlap, path+[choice], new_overlap)
                if flag:
                    break
            else:
                ans = path+[choice]
                if new_overlap <= max_overlap:
                    flag = True
                    break
        return ans, flag

    def _wordpiece_merge(self, embedding, tokens):
        merge_begin = -1
        tokens_select = []
        embed_select = []
        tokens.append('')
        for rank, i in enumerate(tokens):
            if merge_begin >= 0 and "##" not in i:
                # get average embdding when merge wordpiece
                embedding[merge_begin] = torch.sum(
                    embedding[merge_begin:rank], dim=0)/(rank-merge_begin)
                tokens_select.append(self.tokenizer.convert_tokens_to_string(
                    tokens[merge_begin:rank]))
                merge_begin = -1

            if merge_begin < 0 and rank < len(tokens)-1:
                embed_select.append(rank)
                if "##" in tokens[rank+1]:
                    merge_begin = rank
                else:
                    tokens_select.append(i)

        result_embedding = torch.index_select(embedding, 0, torch.LongTensor(
            embed_select).to(self.args.cuda_device))

        return result_embedding, tokens_select

    def train_relation_extraction_model(self):
        dataset = RelExDataset(self.args.triples_tsv, self.tokenizer,
                               self.args.cuda_device, self.args.max_length)
        train_data, test_data = random_split(
            dataset, [len(dataset)-len(dataset)//5, len(dataset)//5])
        train_loader = DataLoader(
            train_data, batch_size=self.args.batch_size, collate_fn=my_collate)
        test_loader = DataLoader(
            test_data, batch_size=self.args.batch_size, collate_fn=my_collate)

        # model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(self.args.cuda_device)
        model = BertForQuestionAnswering.from_pretrained(BERT_MODEL).to(self.args.cuda_device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        best_loss = 0x3fffffff
        best_epoch = 0
        
        if not os.path.exists(self.args.out_folder):
            os.makedirs(self.args.out_folder, exist_ok=True)

        def test(best_loss, best_epoch):
            print("begin test")
            loss = 0
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    if batch:
                        input_ids, token_type_ids, attention_mask, start_positions, end_positions = batch
                    else:
                        continue
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask, start_positions=start_positions.squeeze(-1), end_positions=end_positions.squeeze(-1))
                    loss += outputs.loss.item()
                print("test loss:")
                if best_loss > loss:
                    best_loss = loss
                    best_epoch = epoch
                    # torch.save(model.state_dict(), os.path.join(
                    #     self.args.out_folder, "best_re.pth"))
                    torch.save(model.state_dict(), self.args.model_path)
                return best_loss, best_epoch

        cnt = 1
        for epoch in range(self.args.epoch):
            with tqdm(total=len(train_loader), desc="training BERT-QA for predicate extraction:") as t:
                for batch in train_loader:
                    if batch:
                        input_ids, token_type_ids, attention_mask, start_positions, end_positions = batch
                    else:
                        t.update(1)
                        continue
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask, start_positions=start_positions.squeeze(-1), end_positions=end_positions.squeeze(-1))
                    loss = outputs.loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    t.set_description("Epoch {}:".format(epoch))
                    t.set_postfix(loss="{}".format(loss.item()))
                    t.update(1)

                    if cnt % 1000 == 0:
                        best_loss, best_epoch = test(best_loss, best_epoch)
                    cnt += 1
                best_loss, best_epoch = test(best_loss, best_epoch)

        print("best loss:{:.2f} @epoch {}".format(best_loss, best_epoch))

    def search_entities(self):
        # 1. cos similarity
        # 2. parse Tree complement
        cnt = 0
        if not os.path.exists(self.args.out_folder):
            os.mkdir(self.args.out_folder)
        
        _path = self.ent_result
        if os.path.exists(_path):
            return

        out_tsv = open(_path, "w")
        # sentence_mapping_tsv = open(os.path.join(
        #     self.args.out_folder, self.args.out_file_prefix + "refine_mapping.tsv"), "w")
        sentence_mapping_tsv = open(self.refine_mapping, "w")
        
        print("loading model ...")
        # model = BertModel.from_pretrained("bert-base-uncased").to(self.args.cuda_device)
        model = BertModel.from_pretrained(BERT_MODEL).to(self.args.cuda_device)

        def calculate_embeddings(sent):
            # bert embedding
            encoded_sent = self.tokenizer(
                sent, return_tensors='pt').to(self.args.cuda_device)
            with torch.no_grad():
                output_sent = model(
                    **encoded_sent).last_hidden_state[0][1:-1]
            sent_tokens: list = self.tokenizer.convert_ids_to_tokens(
                encoded_sent['input_ids'][0].tolist()[1:-1])

            # merged custom embedding
            # get average embdding when merge wordpiece
            # for parseTree tokenizer
            return self._wordpiece_merge(output_sent, sent_tokens)
        
        for index in tqdm(range(len(self.gen_sents)), desc="Recovering arguments:"):
            original = self.source_sents[index].strip()
            refined_gen = []
            for line in self.gen_sents[index]:
                gen = refine_gen_sentence(line)
                if gen is None:
                    continue
                if gen not in refined_gen:
                    sentence_mapping_tsv.write(
                        f"{line.strip().split(' <sep> ')[1]}\t{gen}\n")
                    refined_gen.append(gen)

            if refined_gen:
                output_ori, ori_tokens = calculate_embeddings(original)

                # get the ent/rel position (head, tail) in original sentence.
                entity_pos = {}
                # init
                for sent_items in self.entities[index]:
                    if len(sent_items) >= 3:
                        for i in range(len(sent_items)):
                            if sent_items[i]:
                                # record it is a entity or not
                                # entity will use the parseTree
                                entity_pos[sent_items[i]] = i != 0 and i != 2
                remove = []
                # find
                for name in entity_pos:
                    tokens = self.tokenizer.convert_ids_to_tokens(
                        self.tokenizer.encode(name)[1:-1])
                    tokens = self.tokenizer.convert_tokens_to_string(
                        tokens).strip().split(" ")
                    if len(tokens) == 0:
                        remove.append(name)
                        continue
                    head = get_head_index(ori_tokens, tokens)
                    if head is None:
                        remove.append(name)
                        continue
                    tail = min(head + len(tokens), len(ori_tokens))
                    entity_pos[name] = (head, tail, entity_pos[name])
                for i in remove:
                    del entity_pos[i]

            for gen in refined_gen:
                output_gen, gen_tokens = calculate_embeddings(gen)

                # for every ent/rel token in original sentence, calculate token-wise similarity to generated sentence
                similarity = [None for i in range(len(ori_tokens))]
                for name in entity_pos:
                    for i in range(entity_pos[name][0], entity_pos[name][1]):
                        if similarity[i] is not None:
                            continue
                        similarity[i] = cosine_similarity(
                            output_ori[i].unsqueeze(0), output_gen)
                        similarity[i][similarity[i] < self.args.threshold] = 0

                # for every ent/rel calculate entity-wise similarity to generated sentence
                entities_similarity = {}
                for name in entity_pos:
                    head = entity_pos[name][0]
                    tail = entity_pos[name][1]
                    entities_similarity[name] = torch.sum(
                        torch.stack(similarity[head:tail]), dim=0)

                # get candidate ent/rel span in generated sentence
                entities_spans = {}
                tree = self.client.annotate(
                    self.tokenizer.convert_tokens_to_string(gen_tokens)).sentence[0].parseTree
                for name in entity_pos:
                    head = entity_pos[name][0]
                    tail = entity_pos[name][1]
                    not_entity = entity_pos[name][2]
                    temp = self._get_spans(
                        name, gen_tokens, entities_similarity[name], tail-head, tree, self.args.span_num, not_entity, self.args.bert_sim_only)
                    if len(temp) > 0:
                        entities_spans[name] = temp

                # search spans for the best result
                visited = []
                for sent_items in self.entities[index]:
                    temp = []
                    not_find = False
                    for item in sent_items:
                        if item not in entities_spans:
                            not_find = True
                            break
                    if not not_find:
                        self.search_space = []
                        for item in sent_items:
                            self.search_space.append(entities_spans[item])
                        ans, flag = self._search_best_triple(
                            0, max_overlap=self.args.max_overlap_ratio*len(gen_tokens))
                        if flag:
                            heads = []
                            for i in ans:
                                name = self.tokenizer.convert_tokens_to_string(
                                    gen_tokens[i[1]:i[2]]).strip()
                                temp.append(name)
                                heads.append(i[1])
                            if len(temp) >= 3:
                                # swap
                                if heads[0] > heads[2]:
                                    temp[0], temp[2] = temp[2], temp[0]

                                if [temp[0], sent_items[1], temp[2]] not in visited:
                                    visited.append(
                                        [temp[0], sent_items[1], temp[2]])
                                    out_tsv.write(
                                        "{}\t<arg1> {} </arg1> <rel> {} </rel> <arg2> {} </arg2>\n".format(gen, temp[0], temp[1], temp[2]))
                                cnt += 1

    def search_relation(self):
        _path = self.final_result
        # model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(self.args.cuda_device)
        model = BertForQuestionAnswering.from_pretrained(BERT_MODEL).to(self.args.cuda_device)
        if self.args.model_path and self.args.model_path != 'default':
            model.load_state_dict(torch.load(
                self.args.model_path))
        else:
            model.load_state_dict(torch.load(
                os.path.join(self.args.out_folder, "best_re.pth")))
        pattern = re.compile(
            r'(.*?)\t<arg1> (.*?) </arg1> <rel> (.*?) </rel> <arg2> (.*?) </arg2>')
        # out_tsv = open(os.path.join(self.args.out_folder, self.args.out_file_prefix + "result.tsv"), "w")
        out_tsv = open(_path, "w")
        # with open(os.path.join(self.args.out_folder, self.args.out_file_prefix + "result_only_ent.tsv")) as f:
        with open(self.ent_result) as f:
            for line in tqdm(f.readlines(), desc="Recovering predicates:"):
                result = pattern.match(line)
                if result:
                    sent, ent1, ori_rel, ent2 = result.groups()
                    sent = sent.strip()
                    # query =  "ent1" and "ent2"
                    # inputs = [CLS] "ent1" and "ent2" [SEP] sent.
                    question = "\"" + ent1+"\" and \""+ent2+"\""
                    inputs = self.tokenizer(question, sent, max_length=self.args.max_length, return_tensors='pt',
                                            padding='max_length', truncation=True).to(self.args.cuda_device)
                    outputs = model(**inputs)
                    answer_start_index = outputs.start_logits.argmax()
                    answer_end_index = outputs.end_logits.argmax()
                    predict_answer_tokens = inputs.input_ids[0,
                                                             answer_start_index: answer_end_index + 1]
                    rel = self.tokenizer.decode(predict_answer_tokens).strip()
                    if rel and len(rel) < max(15, 3*len(ori_rel.strip())):
                        # if rel:
                        self.build_dataset(out_tsv, sent, ent1, rel, ent2, 0)
                else:
                    print(line)

    def build_dataset(self, f, sent, ent1, rel, ent2, score=0):
        f.write(
            "{}\t<arg1> {} </arg1> <rel> {} </rel> <arg2> {} </arg2>\t{}\n".format(sent, ent1, rel, ent2, 0))


        
        
        

class RelExDataset(Dataset):
    def __init__(self, data_path, tokenizer: BertTokenizer, device, max_length=256) -> None:
        super().__init__()
        with open(data_path) as f:
            self.lines = f.readlines()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def parse_line(self, line):
        pattern = re.compile(
            r".*?<arg1> (.*?) </arg1> <rel> (.*?) </rel> <arg2> (.*?) </arg2>.*?")
        sent, triple, _ = line.split("\t")
        result = pattern.match(triple)
        if not result:
            return None
        ent1, rel, ent2 = result.groups()
        return (sent, ent1, rel, ent2)

    def __getitem__(self, index):
        line = self.lines[index]
        result = self.parse_line(line)
        if not result:
            return None
        sent, ent1, rel, ent2 = result

        question = "\"" + ent1+"\" and \""+ent2+"\""

        target = self.tokenizer(question, sent, max_length=self.max_length, return_tensors='pt',
                                padding='max_length', truncation=True).to(self.device)
        encoded_target = target['input_ids'][0].tolist()

        encoded_rel = self.tokenizer(rel, return_tensors='pt')[
            'input_ids'][0].tolist()[1:-1]
        if len(encoded_rel) == 0:
            return None
        if rel not in ['is', 'are', 'was', 'were']:
            rel_index = get_head_index(encoded_target, encoded_rel)
            if not rel_index:
                return None
            else:
                head = rel_index
            tail = min(head+len(encoded_rel)-1, len(encoded_target))
        else:
            encoded_ent1 = self.tokenizer(ent1, return_tensors='pt')[
                'input_ids'][0].tolist()[1:-1]
            encoded_ent2 = self.tokenizer(ent2, return_tensors='pt')[
                'input_ids'][0].tolist()[1:-1]
            try:
                ent1_head = get_head_index(
                    encoded_target, encoded_ent1, return_all=True)[1]
                ent2_head = get_head_index(
                    encoded_target, encoded_ent2, return_all=True)[1]
            except Exception as e:
                return None

            rel_index = get_head_index(
                encoded_target, encoded_rel, return_all=True)
            head = -1
            if not rel_index:
                return None
            for i in rel_index:
                if i > ent1_head and i < ent2_head:
                    head = i
                    break
            if head == -1:
                return None
            tail = min(head+len(encoded_rel)-1, ent2_head-1)
        return target['input_ids'][0], target['token_type_ids'][0], target['attention_mask'][0], torch.LongTensor([head]).to(self.device), torch.LongTensor([tail]).to(self.device)


        

def refine_gen_sentence(line):
    gen = line.strip().split(" <sep> ")
    if len(gen) < 2:
        return None
    else:
        gen = contractions.fix(gen[1].replace(
            " \'s", "\'s").replace(" n\'t", "n\'t")).strip()
        if gen[-1] == ".":
            gen = gen[:-1].strip()
    return gen


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


# Complement based on syntactic tree
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


def bottom_up_dp_lcs(sent_tokens, entity_tokens):
    """
    longest common subsequence of sent_tokens and entity_tokens
    """
    if len(sent_tokens) == 0 or len(entity_tokens) == 0:
        return []
    dp = [[0 for _ in range(len(entity_tokens) + 1)]
          for _ in range(len(sent_tokens) + 1)]
    for i in range(1, len(sent_tokens) + 1):
        for j in range(1, len(entity_tokens) + 1):
            if sent_tokens[i-1] == entity_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max([dp[i-1][j], dp[i][j-1]])
    # outputs largest common sub-sequence
    i, j = len(sent_tokens), len(entity_tokens)
    index = []
    while i > 0 and j > 0:
        if sent_tokens[i-1] == entity_tokens[j-1] and dp[i][j] == dp[i-1][j-1] + 1:
            index = [i-1] + index
            i, j = i-1, j-1
            continue
        if dp[i][j] == dp[i-1][j]:
            i, j = i-1, j
            continue
        if dp[i][j] == dp[i][j-1]:
            i, j = i, j-1
            continue
    if index:
        return index
    else:
        return [-1]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--triples-tsv', type=str)
    parser.add_argument('--para-source', type=str)
    parser.add_argument('--para-tgt', type=str)

    parser.add_argument('--out-folder', type=str, default="outputs")
    parser.add_argument('--out-file-prefix', type=str, default="")
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--span-num', type=int, default=5)
    parser.add_argument('--max-overlap-ratio', type=float, default=0)
    parser.add_argument('--cuda-device', type=int, default=3)

    parser.add_argument('--max-length', type=int, default=256)

    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=3)
    
    # parser.add_argument('--bert-sim-only', action="store_true") # choose restoration methods
    parser.add_argument('--bert-sim-only', type=str, default="true") # choose restoration methods
    parser.add_argument('--tsv-format', type=str, default="openie4")
    args = parser.parse_args()

    if args.bert_sim_only in ["true", "True"]:
        args.bert_sim_only=True
    else:
        args.bert_sim_only=False

    if "carb" in args.triples_tsv:
        args.tsv_format = "carb"

    sententce_filter = SentenceFilter(args)
    sententce_filter.init_filter()
    sententce_filter.search_entities()
    
    # if not args.bert_sim_only:
    #     if not os.path.exists(args.model_path):
    #         sententce_filter.train_relation_extraction_model()
    #     sententce_filter.search_relation()

    if not os.path.exists(args.model_path):
        sententce_filter.train_relation_extraction_model()
    sententce_filter.search_relation()