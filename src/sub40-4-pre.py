#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import sys,os
import time
import numpy as np
import torch as pt
import argparse
import warnings

from torch import nn, optim
from torch.autograd import Variable
from scipy.spatial.distance import pdist, squareform

from utils import readpickle, savepickle
import torch.nn.functional as F

import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
from tqdm import tqdm
import json


class2idx = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/class2idx_sub40.pkl')
fold2idx = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/fold2idx_sub40.pkl')
lab2idx = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/lab2idx_sub40.pkl')
idx2lab = {value: key for key, value in lab2idx.items()}
sid2label = readpickle('/cmach-data/liuyongchang/protein_sfc/contactlib/save/sid2label.pkl')


tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=device)

def sequencechange(sequence):
    '''
    input a protien sequence and return a sequence with blank intervals
    :param sequence:eg: "MSPLNQ"
    :return: eg: "M S P L N Q"
    '''
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


dataset_split_json = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/dataset_split_sub40.pkl')
train_file = dataset_split_json['train_file']
val_file = dataset_split_json['val_file']
test_file = dataset_split_json['test_file']


print('processing training set')
predict_train, str_id_train = [], []
for ii in tqdm(range(len(train_file))):
    fn = train_file[ii]
    # print(ii, fn)
    sid = os.path.basename(fn)[:-5]
    sequence = readpickle('/cmach-data/liuyongchang/protein_sfc/seq/dataset/{}.pkl'.format(sid))
    seq_len = len(sequence)
    # sequence = sequencechange(sequence)
    # sequences_Example = [sequence]
    # sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    # embedding = fe(sequences_Example)
    # embedding = np.array(embedding)
    # embedding = embedding[0,0,:,:]
    # start_Idx = 1
    # end_Idx = seq_len+1
    # embedding = embedding[start_Idx:end_Idx]

    # with open(fn, 'r') as f: scop = json.load(f)
    # model = scop['model']
    # size = len(model) # number of residues
    # coord = np.ones((size, 3), dtype=np.float32) * np.inf
    # for i in model.keys():
    #     ii = int(i) - 1
    #     coord[ii, 0] = model[i]['x']
    #     coord[ii, 1] = model[i]['y']
    #     coord[ii, 2] = model[i]['z']
    # dist = np.nan_to_num(squareform(pdist(np.array(coord, dtype=np.float32))), nan=np.inf)
    # coord_info = {"coord": coord, "dist": dist}
    # savepickle(coord_info, '/cmach-data/liuyongchang/protein_sfc/spaT/dataset/{}_coord.pkl'.format(sid))

    pclass = sid2label[sid]['class']
    pclass = class2idx[pclass]
    fold = sid2label[sid]['fold']
    fold = fold2idx[fold]
    label = sid2label[sid]['superfamily']
    label = lab2idx[label]
    # savepickle(embedding, '/cmach-data/liuyongchang/protein_sfc/spaT/dataset_emb/{}.pkl'.format(sid))
    predict_train.append(dict(sid = sid, length = seq_len, label = [label], pclass = [pclass], fold = [fold]))

savepickle(predict_train, '/cmach-data/liuyongchang/protein_sfc/PSST/save/data_train_sub40.pkl')
print('done')



print('processing val set')
predict_val, str_id_val = [], []
for ii in tqdm(range(len(val_file))):
    fn = val_file[ii]
    # print(ii, fn)
    sid = os.path.basename(fn)[:-5]
    sequence = readpickle('/cmach-data/liuyongchang/protein_sfc/seq/dataset/{}.pkl'.format(sid))
    seq_len = len(sequence)
    # sequence = sequencechange(sequence)
    # sequences_Example = [sequence]
    # sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    # embedding = fe(sequences_Example)
    # embedding = np.array(embedding)
    # embedding = embedding[0,0,:,:]
    # start_Idx = 1
    # end_Idx = seq_len+1
    # embedding = embedding[start_Idx:end_Idx]

    # with open(fn, 'r') as f: scop = json.load(f)
    # model = scop['model']
    # size = len(model) # number of residues
    # coord = np.ones((size, 3), dtype=np.float32) * np.inf
    # for i in model.keys():
    #     ii = int(i) - 1
    #     coord[ii, 0] = model[i]['x']
    #     coord[ii, 1] = model[i]['y']
    #     coord[ii, 2] = model[i]['z']
    # dist = np.nan_to_num(squareform(pdist(np.array(coord, dtype=np.float32))), nan=np.inf)
    # coord_info = {"coord": coord, "dist": dist}
    # savepickle(coord_info, '/cmach-data/liuyongchang/protein_sfc/spaT/dataset/{}_coord.pkl'.format(sid))

    pclass = sid2label[sid]['class']
    pclass = class2idx[pclass]
    fold = sid2label[sid]['fold']
    fold = fold2idx[fold]
    label = sid2label[sid]['superfamily']
    label = lab2idx[label]
    # savepickle(embedding, '/cmach-data/liuyongchang/protein_sfc/spaT/dataset_emb/{}.pkl'.format(sid))
    predict_val.append(dict(sid = sid, length = seq_len, label = [label], pclass = [pclass], fold = [fold]))

savepickle(predict_val, '/cmach-data/liuyongchang/protein_sfc/PSST/save/data_val_sub40.pkl')
print('done')



print('processing test set')

predict_test, str_id_test = [], []
for ii in tqdm(range(len(test_file))):
    fn = test_file[ii]
    # print(ii, fn)
    sid = os.path.basename(fn)[:-5]
    sequence = readpickle('/cmach-data/liuyongchang/protein_sfc/seq/dataset/{}.pkl'.format(sid))
    seq_len = len(sequence)
    # sequence = sequencechange(sequence)
    # sequences_Example = [sequence]
    # sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    # embedding = fe(sequences_Example)
    # embedding = np.array(embedding)
    # embedding = embedding[0,0,:,:]
    # start_Idx = 1
    # end_Idx = seq_len+1
    # embedding = embedding[start_Idx:end_Idx]

    # with open(fn, 'r') as f: scop = json.load(f)
    # model = scop['model']
    # size = len(model) # number of residues
    # coord = np.ones((size, 3), dtype=np.float32) * np.inf
    # for i in model.keys():
    #     ii = int(i) - 1
    #     coord[ii, 0] = model[i]['x']
    #     coord[ii, 1] = model[i]['y']
    #     coord[ii, 2] = model[i]['z']
    # dist = np.nan_to_num(squareform(pdist(np.array(coord, dtype=np.float32))), nan=np.inf)
    # coord_info = {"coord": coord, "dist": dist}
    # savepickle(coord_info, '/cmach-data/liuyongchang/protein_sfc/spaT/dataset/{}_coord.pkl'.format(sid))

    pclass = sid2label[sid]['class']
    pclass = class2idx[pclass]
    fold = sid2label[sid]['fold']
    fold = fold2idx[fold]
    label = sid2label[sid]['superfamily']
    label = lab2idx[label]
    # savepickle(embedding, '/cmach-data/liuyongchang/protein_sfc/spaT/dataset_emb/{}.pkl'.format(sid))
    predict_test.append(dict(sid = sid, length = seq_len, label = [label], pclass = [pclass], fold = [fold]))

savepickle(predict_test, '/cmach-data/liuyongchang/protein_sfc/PSST/save/data_test_sub40.pkl')
print('done')
