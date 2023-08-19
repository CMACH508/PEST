
import sys,os
import time
import numpy as np
import argparse
import warnings
from torch import nn, optim
from torch.autograd import Variable
from scipy.spatial.distance import pdist, squareform
from utils import readpickle, savepickle
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from transformers import logging
logging.set_verbosity_error()
import re
from tqdm import tqdm
import json


class2idx = readpickle('../save/class2idx.pkl')
fold2idx = readpickle('../save/fold2idx.pkl')
lab2idx = readpickle('../save/lab2idx.pkl')
idx2lab = {value: key for key, value in lab2idx.items()}
sid2label = readpickle('../save/sid2label.pkl')


tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=device)


def sequencechange(sequence):
    new_seq = ""
    count = 0
    for i in sequence:
        if i == ' ':
            continue
        new_seq += i
        count += 1
        if count == len(sequence):
            continue
        new_seq += ' '
    return new_seq


test_json_list = ['../dataset/json/d3r1pa_.json', '../dataset/json/d6jl3a_.json', '../dataset/json/d7ndyg_.json']
print('processing test set')
data_test = []
for i in tqdm(range(len(test_json_list))):
    fn = test_json_list[i]
    sid = os.path.splitext(os.path.basename(fn))[0]

    with open(fn, 'r') as f: scop = json.load(f)
    model = scop['model']
    sequence = ""
    for idx, res in model.items():
        AA = res["res"]
        sequence += AA

    seq_len = len(sequence)
    sequence = sequencechange(sequence)
    sequences_Example = [sequence]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    embedding = fe(sequences_Example)
    embedding = np.array(embedding)
    embedding = embedding[0,0,:,:]
    start_Idx = 1
    end_Idx = seq_len+1
    embedding = embedding[start_Idx:end_Idx]

    size = len(model)
    coord = np.ones((size, 3), dtype=np.float32) * np.inf
    for i in model.keys():
        ii = int(i) - 1
        coord[ii, 0] = model[i]['x']
        coord[ii, 1] = model[i]['y']
        coord[ii, 2] = model[i]['z']
    dist = np.nan_to_num(squareform(pdist(np.array(coord, dtype=np.float32))), nan=np.inf)
    dist = {"coord": coord, "dist": dist}
    savepickle(dist, '../dataset/dist/{}_dist.pkl'.format(sid))

    pclass = sid2label[sid]['class']
    pclass = class2idx[pclass]
    fold = sid2label[sid]['fold']
    fold = fold2idx[fold]
    label = sid2label[sid]['superfamily']
    label = lab2idx[label]
    savepickle(embedding, '../dataset/embedding/{}.pkl'.format(sid))
    data_test.append(dict(sid = sid, length = seq_len, label = [label], pclass = [pclass], fold = [fold]))

savepickle(data_test, '../save/data_test.pkl')

