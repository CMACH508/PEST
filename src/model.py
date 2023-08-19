#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import json
import math
import random
import numpy as np
import torch
import torch as pt
import re
from einops import rearrange, repeat
from torch import nn
from torch.autograd import Variable
from glob import glob
from scipy.spatial.distance import pdist, squareform

from utils import readpickle

from transformer import Transformer, Transformer_structure


alphabet_res = 'LAGVESKIDTRPNFQYHMCW'
alphabet_ss = 'HE TSGBI'
radius = 8
diameter = radius * 2 + 1
volume = diameter * (diameter * 2 - 1)
size0, size1, size2 = 7, 1257, 2065 


def listBatch(bucket, keyword, batchsize, batchlen=512):
    result = []
    bucket = sorted(bucket, key=lambda k: k[keyword])
    while (len(bucket) > 0):
        batchsize = min([batchsize, len(bucket)])
        while bucket[batchsize-1][keyword] > batchlen:
            batchsize, batchlen = (batchsize + 1) // 2, batchlen * 2
        result.append(bucket[:batchsize])
        bucket = bucket[batchsize:]
    random.shuffle(result)
    return result


def dataloader(data, batchsize):
    for batch in listBatch(data, 'length', batchsize):
        sizemax = batch[-1]['length']
        seq = np.zeros([len(batch), sizemax, 1024], dtype=np.float32)
        dist = np.zeros([len(batch), sizemax, sizemax], dtype=np.float32)
        mask = np.ones([len(batch), sizemax], dtype=bool)
        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = b['length']
            sid = b['sid']
            embedding = readpickle('../dataset/embedding/{}.pkl'.format(sid))
            seq[i, :size] = embedding
            mask[i, :size] = False
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']

            distance_matrix = readpickle('../dataset/dist/{}_dist.pkl'.format(sid))["dist"]
            start_x = 0
            start_y = 0
            end_x = start_x + distance_matrix.shape[0]
            end_y = start_y + distance_matrix.shape[1]
            dist[i, start_x:end_x, start_y:end_y] = distance_matrix
            
        seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
        dist = pt.from_numpy(dist)
        mask = pt.from_numpy(mask)
        pclass = pt.from_numpy(pclass)
        fold = pt.from_numpy(fold)
        label = pt.from_numpy(label)

        yield seq, dist, mask, pclass, fold, label


class BaseNet(nn.Module):
    def __init__(self, width, multitask=True):
        super(BaseNet, self).__init__()

        self.multitask = multitask
        self.out0 = nn.Linear(width, size0)
        self.out1 = nn.Linear(width, size1)
        self.out2 = nn.Linear(width, size2)

    def forward(self, mem):
        if self.multitask: return self.out0(mem), self.out1(mem), self.out2(mem), mem
        else: return None, None, self.out2(mem), mem


class PSST(BaseNet):
    def __init__(self, depth, width, multitask=True):
        super(PSST, self).__init__(width, multitask=multitask)
        assert(width % 64 == 0)
        nhead, ndense = width//64, width*4
        dim_head = width // nhead
        dropout = 0.1
        self.transformer = Transformer_structure(width, depth, nhead, dim_head, ndense, dropout)

    def forward(self, x, dist, mask):
        mem = self.transformer(x, dist, mask).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        return super().forward(mem)

