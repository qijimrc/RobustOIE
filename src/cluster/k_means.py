import os
from sklearn.decomposition import PCA
from typing import Counter
from sklearn import cluster
import json
import numpy
import matplotlib.pyplot as plt
# from random import sample
import random
import argparse
from tqdm import tqdm


def seq2dict(seq, max_level):
    root = {'value': seq.split(" ")[0]}
    if max_level == 1:
        return root
    child = []
    stack = 0
    begin = -1
    for i in range(0, len(seq)):
        if seq[i] == '(':
            if stack == 0:
                begin = i
            stack += 1
        elif seq[i] == ')':
            stack -= 1
            if stack == 0:
                child.append(seq2dict(seq[begin+1:i], max_level-1))
    if child:
        root['child'] = child
    return root


def dict2bag(root, bag):
    bag[tag_dict[root['value']]] += 1
    if 'child' in root:
        for child in root['child']:
            dict2bag(child, bag)


def bags_algorithm(k):
    bags = []
    for line in tqdm(lines):
        parseTree = line.split(' <sep> ')[1].strip()[1:-1]
        root = seq2dict(parseTree, 4)
        bag = [0]*len(tag_dict)
        dict2bag(root, bag)
        bag[0] -= 1
        bag[1] -= 1
        bags.append(bag)

    x = numpy.array(bags)
    features = x/x.sum(1)[:, None]

    model = cluster.KMeans(n_clusters=k, max_iter=1000,
                           n_jobs=4, init="k-means++")
    model.fit(features)
    cnt = Counter(model.labels_)
    files = []
    for i in range(k):
        files.append(open(f"clusters/{i}.txt", "w"))
    for i in range(len(model.labels_)):
        files[model.labels_[i]].write(lines[i])
    for i in range(k):
        files[i].close()

    with open(train_data) as f:
        samples = random.sample(f.readlines(), 20000)

    distances = []
    for i in range(k):
        center = model.cluster_centers_[i]
        temp = 0
        for line in samples:
            parseTree = line.split(' <sep> ')[1].strip()[1:-1]
            root = seq2dict(parseTree, 4)
            bag = [0]*len(tag_dict)
            dict2bag(root, bag)
            bag[0] -= 1
            bag[1] -= 1
            feature = numpy.array(bag)
            temp = numpy.sqrt(((center-feature)**2).sum())
        temp /= len(samples)
        distances.append(temp)
    for i in cnt.most_common():
        print(f'{i[0]}: cnt={i[1]},dis={distances[i[0]]}')


def preorderTraversal(root, nums):
    nums.append(tag_dict[root['value']])
    if 'child' in root:
        for child in root['child']:
            preorderTraversal(child, nums)


def levelorderTraversal(root, nums):
    queue = [root]
    while queue:
        cur = queue.pop(0)
        nums.append(tag_dict[cur['value']])
        if 'child' in cur:
            for child in cur['child']:
                queue.append(child)


def lcs_distance(s1, s2, weight_decay=1):
    tot_len = 0
    dp = [[0]*(len(s2)) for _ in range(len(s1))]
    for i in range(len(s2)):
        if s1[0] == s2[i]:
            dp[0][i] == 1

    for i in range(len(s1)):
        if s1[i] == s2[0]:
            dp[i][0] == 1

    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = 0
                if dp[i-1][j-1] > 1:
                    tot_len += dp[i-1][j-1] * \
                        (weight_decay**i+weight_decay**j)/2
    if dp[len(s1)-1][len(s2)-1] > 1:
        tot_len += dp[len(s1)-1][len(s2)-1]
    return 1-tot_len/min(len(s1), len(s2))


def lcs_distance_qiji(s1, s2, weight_decay=1):
    tot_len = 0
    tot_m = 0
    dp = [[0]*(len(s2)) for _ in range(len(s1))]
    for i in range(len(s2)):
        if s1[0] == s2[i]:
            dp[0][i] = 1

    for i in range(len(s1)):
        if s1[i] == s2[0]:
            dp[i][0] = 1

    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = 0
                if dp[i-1][j-1] > 1:
                    tot_len += dp[i-1][j-1] * (weight_decay**tot_m)
                    tot_m += 1
    if dp[len(s1)-1][len(s2)-1] > 1:
        tot_len += dp[len(s1)-1][len(s2)-1] * (weight_decay**tot_m)
    return 1-tot_len/min(len(s1), len(s2))



def lcs_algorithm(source_files, train_source, k, traversal_algorithm, distance_func, weight_decay=1, name=""):
    
#     with open(test_source) as f:
#         lines = f.readlines()
    lines = []
    for source_f in source_files:
        with open(source_f) as f:
            pre_line = ""
            for line in f:
                if line.split(' <sep> ')[0].strip() != pre_line:
                    lines.append(line)
                pre_line = line.split(' <sep> ')[0].strip()
    print("total test lines: %d" % len(lines))
        
    num_seqs = []
    for line in tqdm(lines):
        parseTree = line.split(' <sep> ')[1].strip()[1:-1]
        root = seq2dict(parseTree, 4)
        seq = []
        traversal_algorithm(root, seq)
        num_seqs.append(seq)
    k_means = random.sample(num_seqs, 300)
    for epoch in range(100):
        print(f"epoch:{epoch}")
        k_sets = [[] for i in range(k)]
        for i in tqdm(range(len(num_seqs))):
            min_distance = 1
            label = -1
            for j in range(k):
                distance = distance_func(
                    k_means[j][1:], num_seqs[i][1:], weight_decay)
                if distance < min_distance:
                    min_distance = distance
                    label = j
            k_sets[label].append(i)
        for i in numpy.argsort([len(t) for t in k_sets]):
            print(f'{i}: cnt={len(k_sets[i])}')
        for i in range(k):
            min_tot_distance = len(k_sets[i])
            print(f"calulate {i}_means")
            for j in range(len(k_sets[i])):
                tot_distance = 0
                for j2 in range(len(k_sets[i])):
                    tot_distance += distance_func(
                        num_seqs[k_sets[i][j]][1:], num_seqs[k_sets[i][j2]][1:])
                if tot_distance < min_tot_distance:
                    min_tot_distance = tot_distance
                    k_means[i] = num_seqs[k_sets[i][j]]

    distances = evaluate(train_source, k_means, k, traversal_algorithm, distance_func)
    if not os.path.exists(f'{name}'):
        os.mkdir(f'{name}')
    means_out = open(f"{name}/kmeans.txt", "w")
    results_out = open(f"{name}/results.txt", "w")
    for i in numpy.argsort([len(t) for t in k_sets]):
        print(f'{i}: cnt={len(k_sets[i])},dis={distances[i]}')
        means_out.write(json.dumps(k_means[i])+"\n")
        results_out.write(f'{i}: cnt={len(k_sets[i])},dis={distances[i]}\n')
        with open(f"{name}/{i}.txt", 'w') as f:
            for j in k_sets[i]:
                f.write(lines[j])


def evaluate(train_source, k_means, k, traversal_algorithm, distance_func):
    distances = []
    with open(train_source) as f:
        samples = random.sample(f.readlines(), 20000)
    for i in range(k):
        temp = 0
        for line in samples:
            parseTree = line.split(' <sep> ')[1].strip()[1:-1]
            root = seq2dict(parseTree, 4)
            seq = []
            traversal_algorithm(root, seq)
            temp += distance_func(k_means[i][1:], seq[1:])
        distances.append(temp/len(samples))
    return distances


def copy_result(k, name, extraction_files, target_dir):
    golden_tri = {}
    for ext_f in extraction_files:
        with open(ext_f) as f:
            for line in f:
                result = line.split("\t")[0].strip()
                if result in golden_tri:
                    golden_tri[result] += line
                else:
                    golden_tri[result] = line

    for i in range(k):
        s = ""
        out_sentences = open(
            target_dir+f"/{name}_{i}_sentences.txt", "w")
        if not os.path.exists(target_dir+f"/{name}_{i}"):
            os.mkdir(target_dir+f"/{name}_{i}")
        out_tsv = open(target_dir+f"/{name}_{i}/extractions.tsv", "w")
        with open(f"{name}/{i}.txt") as f:
            for line in f:
                if s != line.split("<sep>")[0].strip():
                    s = line.split("<sep>")[0].strip()
                    if s in golden_tri:
                        out_tsv.write(golden_tri[s])
                        out_sentences.write(s+'\n')
        out_sentences.close()
        out_tsv.close()




if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--carb_dev_src", type=str)
    parser.add_argument("--carb_test_src", type=str)
    parser.add_argument("--train_source", type=str)
    parser.add_argument("--carb_dev_ext", type=str)
    parser.add_argument("--carb_test_ext", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--tag_dict", type=str)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    with open(args.tag_dict) as f:
        tag_dict = json.load(f)

#     # bags_algorithm(k)
    source_files = [args.carb_dev_src, args.carb_test_src]
    extractions_files = [args.carb_dev_ext, args.carb_test_ext]
    lcs_algorithm(source_files, args.train_source, args.k, levelorderTraversal, lcs_distance_qiji, 0.95, 'level')
    copy_result(args.k, 'level', extractions_files, args.target_dir)