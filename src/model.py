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
size0, size1, size2 = 7, 1257, 2065 # sub40


def listBatch_contact(bucket, keyword, batchsize, batchlen=512):
    result = []
    bucket = sorted(bucket, key=lambda k: len(k[keyword]))
    while (len(bucket) > 0):
        batchsize = min([batchsize, len(bucket)])
        while len(bucket[batchsize-1][keyword]) > batchlen:
            batchsize, batchlen = (batchsize + 1) // 2, batchlen * 2
        result.append(bucket[:batchsize])
        bucket = bucket[batchsize:]
    random.shuffle(result)
    return result


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


def dataloader_contact(data, batchsize):
    for batch in listBatch_contact(data, 'hbond', batchsize):
        sizemax = len(batch[-1]['hbond'])

        seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
        mask = np.ones([len(batch), sizemax], dtype=bool)
        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = len(b['hbond'])
            # print(size)

            seq[i, :size] = np.array([pdist(b['hbond'][j]['coord']) for j in range(size)])
            mask[i, :size] = False
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']
            
        seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
        mask = pt.from_numpy(mask)
        pclass = pt.from_numpy(pclass)
        fold = pt.from_numpy(fold)
        label = pt.from_numpy(label)

        yield seq, mask, pclass, fold, label


def dataloader_contact_test(data, batchsize):
    for batch in listBatch_contact(data, 'hbond', batchsize):
        sids = []
        sizemax = len(batch[-1]['hbond'])

        seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
        mask = np.ones([len(batch), sizemax], dtype=bool)
        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = len(b['hbond'])
            sid = b['sid']
            sids.append(sid)
            # print(size)

            seq[i, :size] = np.array([pdist(b['hbond'][j]['coord']) for j in range(size)])
            mask[i, :size] = False
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']
            
        seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
        mask = pt.from_numpy(mask)
        pclass = pt.from_numpy(pclass)
        fold = pt.from_numpy(fold)
        label = pt.from_numpy(label)

        yield seq, mask, pclass, fold, label, sids


def iterTrainBond(data, batchsize, bucketsize, noiserate=0.1):
    for batch in listBatch_contact(random.sample(data, batchsize*bucketsize), 'hbond', batchsize):
        sizemax = len(batch[-1]['hbond'])
        noise = np.random.normal(0, noiserate, [len(batch), sizemax, diameter*2, 3])

        seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
        mask = np.ones([len(batch), sizemax], dtype=bool)
        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = len(b['hbond'])

            seq[i, :size] = np.array([pdist(b['hbond'][j]['coord'] + noise[i, j]) for j in range(size)])
            mask[i, :size] = False
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']
            
        seq[seq < 1.0] = 0.0
        seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
        mask = pt.from_numpy(mask)
        pclass = pt.from_numpy(pclass)
        fold = pt.from_numpy(fold)
        label = pt.from_numpy(label)

        yield seq, mask, pclass, fold, label


def iterTestBond_test(data, batchsize):
    for batch in listBatch_contact(data, 'hbond', batchsize):
        sids = []
        sizemax = len(batch[-1]['hbond'])

        seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
        mask = np.ones([len(batch), sizemax], dtype=bool)
        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = len(b['hbond'])
            sid = b['sid']
            sids.append(sid)

            seq[i, :size] = np.array([pdist(b['hbond'][j]['coord']) for j in range(size)])
            mask[i, :size] = False
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']
            
        seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
        mask = pt.from_numpy(mask)
        pclass = pt.from_numpy(pclass)
        fold = pt.from_numpy(fold)
        label = pt.from_numpy(label)

        yield seq, mask, pclass, fold, label, sids


def iterTestBond(data, batchsize):
    for batch in listBatch_contact(data, 'hbond', batchsize):
        sizemax = len(batch[-1]['hbond'])

        seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
        mask = np.ones([len(batch), sizemax], dtype=bool)
        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = len(b['hbond'])
            # print(size)

            seq[i, :size] = np.array([pdist(b['hbond'][j]['coord']) for j in range(size)])
            mask[i, :size] = False
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']
            
        seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
        mask = pt.from_numpy(mask)
        pclass = pt.from_numpy(pclass)
        fold = pt.from_numpy(fold)
        label = pt.from_numpy(label)

        yield seq, mask, pclass, fold, label


def dataloader_deepfold(data, batchsize):
    for batch in listBatch(data, 'length', batchsize):
        sizemax = batch[-1]['length']
        dist = np.zeros([len(batch), 1, sizemax, sizemax], dtype=np.float32)
        mask = np.zeros([len(batch)], dtype=int)

        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = b['length']
            sid = b['sid']
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']
            mask[i] = size

            distance_matrix = readpickle('/cmach-data/liuyongchang/protein_sfc/spaT/dataset/{}_coord.pkl'.format(sid))["dist"]
            start_x = 0
            start_y = 0
            end_x = start_x + distance_matrix.shape[0]
            end_y = start_y + distance_matrix.shape[1]
            dist[i, 0, start_x:end_x, start_y:end_y] = distance_matrix
            
        dist = pt.from_numpy(dist)
        mask = pt.from_numpy(mask)
        pclass = pt.from_numpy(pclass)
        fold = pt.from_numpy(fold)
        label = pt.from_numpy(label)

        yield dist, mask, pclass, fold, label


def dataloader_deepfold_test(data, batchsize):
    for batch in listBatch(data, 'length', batchsize):
        sids = []
        lengths = []
        sizemax = batch[-1]['length']
        dist = np.zeros([len(batch), 1, sizemax, sizemax], dtype=np.float32)
        mask = np.zeros([len(batch)], dtype=int)

        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = b['length']
            lengths.append(size)
            sid = b['sid']
            sids.append(sid)
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']
            mask[i] = size

            distance_matrix = readpickle('/cmach-data/liuyongchang/protein_sfc/spaT/dataset/{}_coord.pkl'.format(sid))["dist"]
            start_x = 0
            start_y = 0
            end_x = start_x + distance_matrix.shape[0]
            end_y = start_y + distance_matrix.shape[1]
            dist[i, 0, start_x:end_x, start_y:end_y] = distance_matrix
            
        dist = pt.from_numpy(dist)
        mask = pt.from_numpy(mask)
        pclass = pt.from_numpy(pclass)
        fold = pt.from_numpy(fold)
        label = pt.from_numpy(label)

        yield dist, mask, pclass, fold, label, sids, lengths


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
            embedding = readpickle('/cmach-data/liuyongchang/protein_sfc/spaT/dataset_emb/{}.pkl'.format(sid))
            seq[i, :size] = embedding
            mask[i, :size] = False
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']

            distance_matrix = readpickle('/cmach-data/liuyongchang/protein_sfc/spaT/dataset/{}_coord.pkl'.format(sid))["dist"]
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


def dataloader_test(data, batchsize):
    for batch in listBatch(data, 'length', batchsize):
        sids = []
        lengths = []
        sizemax = batch[-1]['length']
        seq = np.zeros([len(batch), sizemax, 1024], dtype=np.float32)
        dist = np.zeros([len(batch), sizemax, sizemax], dtype=np.float32)
        mask = np.ones([len(batch), sizemax], dtype=bool)
        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = b['length']
            lengths.append(size)
            sid = b['sid']
            sids.append(sid)
            embedding = readpickle('/cmach-data/liuyongchang/protein_sfc/spaT/dataset_emb/{}.pkl'.format(sid))
            seq[i, :size] = embedding
            mask[i, :size] = False
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']

            distance_matrix = readpickle('/cmach-data/liuyongchang/protein_sfc/spaT/dataset/{}_coord.pkl'.format(sid))["dist"]
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

        yield seq, dist, mask, pclass, fold, label, sids, lengths


amino_acid_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acid_alphabet)}

def encoding(sequence, alphabet):
    # sequence = re.sub(r"[UZOB]", "X", sequence)
    sequence = ''.join([aa if aa in amino_acid_alphabet else 'X' for aa in sequence])
    encoding = np.zeros((len(sequence), len(alphabet)))
    encoding_ = np.zeros(len(sequence))
    for i, aa in enumerate(sequence):
        encoding[i, amino_acid_to_index[aa]] = 1
        encoding_[i] = amino_acid_to_index[aa]
    return encoding, encoding_

def dataloader_wo_pretrain(data, batchsize):
    for batch in listBatch(data, 'length', batchsize):
        sizemax = batch[-1]['length']
        seq = np.zeros([len(batch), sizemax], dtype=np.float32)
        dist = np.zeros([len(batch), sizemax, sizemax], dtype=np.float32)
        mask = np.ones([len(batch), sizemax], dtype=bool)
        pclass = np.zeros([len(batch), len(batch[-1]['pclass'])], dtype=np.int64)
        fold = np.zeros([len(batch), len(batch[-1]['fold'])], dtype=np.int64)
        label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
        for i, b in enumerate(batch):
            size = b['length']
            sid = b['sid']
            sequence = readpickle('/cmach-data/liuyongchang/protein_sfc/seq/dataset/{}.pkl'.format(sid))
            # embedding = readpickle('/cmach-data/liuyongchang/protein_sfc/spaT/dataset_emb/{}.pkl'.format(sid))
            _, embedding = encoding(sequence, amino_acid_alphabet)
            seq[i, :size] = embedding
            mask[i, :size] = False
            pclass[i] = b['pclass']
            fold[i] = b['fold']
            label[i] = b['label']

            distance_matrix = readpickle('/cmach-data/liuyongchang/protein_sfc/spaT/dataset/{}_coord.pkl'.format(sid))["dist"]
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


class DistBlock(nn.Module):
    def __init__(self, dim):
        super(DistBlock, self).__init__()

        self.dim = dim

    def forward(self, x):
        x1 = x / 3.8; x2 = x1 * x1; x3 = x2 * x1
        xx = pt.cat([1/(1+x1), 1/(1+x2), 1/(1+x3)], dim=self.dim).cuda()
        return xx


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


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(cin, cout, kernel_size=5, stride=2, padding=2)
        self.norm = nn.LayerNorm(cout)
        self.act = nn.ModuleList([nn.Dropout2d(dropout), nn.ReLU()])

    def forward(self, x):
        x = self.norm(self.conv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for m in self.act:
            x = m(x)
        return x


class DeepFold(nn.Module):
    def __init__(self, width):
        super(DeepFold, self).__init__()

        self.embed = DistBlock(1)

        cdim = [3, 64, 128, 256, 512, 512, width]
        conv = [ConvBlock(cin, cout) for cin, cout in zip(cdim[:-1], cdim[1:])]
        self.conv = nn.ModuleList(conv)

        self.out2 = nn.Linear(cdim[-1], size2)

    def masked_fill_(self, data, size):
        m = pt.ones([data.size(0), 1, data.size(-1)]).bool().cuda()
        for i, s in enumerate(size):
            m[i, :, :s] = False
        return data.masked_fill(m.unsqueeze(2), 0).masked_fill(m.unsqueeze(3), 0)

    def forward(self, x, size):
        mem = self.masked_fill_(self.embed(x), size)
        for layer in self.conv:
            size = (size + 1) // 2
            mem = self.masked_fill_(layer(mem), size)
        mask = pt.logical_not(pt.eye(mem.size(-1), dtype=pt.bool)).reshape([1, 1, *mem.shape[-2:]]).cuda()
        mem = mem.masked_fill_(mask, 0)
        mem = mem.sum(-1).sum(-1) / size.unsqueeze(1)
        return None, None, self.out2(mem), mem
        

class DeepFold_mt(BaseNet):
    def __init__(self, width, multitask=True):
        super(DeepFold_mt, self).__init__(width, multitask=multitask)

        self.embed = DistBlock(1)

        cdim = [3, 64, 128, 256, 512, 512, width]
        conv = [ConvBlock(cin, cout) for cin, cout in zip(cdim[:-1], cdim[1:])]
        self.conv = nn.ModuleList(conv)

        self.out2 = nn.Linear(cdim[-1], size2)

    def masked_fill_(self, data, size):
        m = pt.ones([data.size(0), 1, data.size(-1)]).bool().cuda()
        for i, s in enumerate(size):
            m[i, :, :s] = False
        return data.masked_fill(m.unsqueeze(2), 0).masked_fill(m.unsqueeze(3), 0)

    def forward(self, x, size):
        mem = self.masked_fill_(self.embed(x), size)
        for layer in self.conv:
            size = (size + 1) // 2
            mem = self.masked_fill_(layer(mem), size)
        mask = pt.logical_not(pt.eye(mem.size(-1), dtype=pt.bool)).reshape([1, 1, *mem.shape[-2:]]).cuda()
        mem = mem.masked_fill_(mask, 0)
        mem = mem.sum(-1).sum(-1) / size.unsqueeze(1)
        return super().forward(mem)


class TransNet(BaseNet):
    def __init__(self, depth, width, multitask=True):
        super(TransNet, self).__init__(width, multitask=multitask)
        assert(width % 64 == 0)
        nhead, ndense = width//64, width*4

        self.embed = nn.Sequential(DistBlock(-1),
                                   nn.Linear(volume*3, ndense), nn.LayerNorm(ndense), nn.ReLU(),
                                   nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())

        layer_encod = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=ndense, dropout=0.1)
        self.encod = nn.TransformerEncoder(layer_encod, depth)

    def forward(self, x, mask):
        mem = self.encod(self.embed(x).permute(1, 0, 2), src_key_padding_mask=mask).permute(1, 0, 2).masked_fill(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        return super().forward(mem)


class SeqNet(BaseNet):
    def __init__(self, depth, width, multitask=True):
        super(SeqNet, self).__init__(width, multitask=multitask)
        assert(width % 64 == 0)
        nhead, ndense = width//64, width*4

        # self.embed = nn.Sequential(DistBlock(-1),
        #                            nn.Linear(volume*3, ndense), nn.LayerNorm(ndense), nn.ReLU(),
        #                            nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())

        # layer_encod = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=ndense, dropout=0.1)
        # self.encod = nn.TransformerEncoder(layer_encod, depth)

        self.pos_embedding = nn.Parameter(torch.randn(1, 3000, width))
        dim_head = width // nhead
        dropout = 0.1
        self.transformer = Transformer(width, depth, nhead, dim_head, ndense, dropout)

    def forward(self, x, mask):
        # mem = self.encod(x.permute(1, 0, 2), src_key_padding_mask=mask).permute(1, 0, 2).masked_fill_(mask.unsqueeze(2), 0)
        # _, n, _ = x.shape
        # x = x + self.pos_embedding[:, :n]
        # x = x.masked_fill_(mask.unsqueeze(2), 0)
        mem = self.transformer(x, mask).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        return super().forward(mem)


class PSST_wo_struct(BaseNet):
    def __init__(self, depth, width, multitask=True):
        super(PSST_wo_struct, self).__init__(width, multitask=multitask)
        assert(width % 64 == 0)
        nhead, ndense = width//64, width*4
        self.pos_embedding = nn.Parameter(torch.randn(1, 1600, width))
        dim_head = width // nhead
        dropout = 0.1
        self.transformer = Transformer(width, depth, nhead, dim_head, ndense, dropout)

    def forward(self, x, mask):
        # mem = self.encod(x.permute(1, 0, 2), src_key_padding_mask=mask).permute(1, 0, 2).masked_fill_(mask.unsqueeze(2), 0)
        _, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = x.masked_fill_(mask.unsqueeze(2), 0)
        mem = self.transformer(x, mask).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        return super().forward(mem)


class PSST_wo_pretrain(BaseNet):
    def __init__(self, depth, width, multitask=True):
        super(PSST_wo_pretrain, self).__init__(width, multitask=multitask)
        assert(width % 64 == 0)
        nhead, ndense = width//64, width*4
        dim_head = width // nhead
        dropout = 0.1
        self.embedding = nn.Embedding(21, 1024)
        self.transformer = Transformer_structure(width, depth, nhead, dim_head, ndense, dropout)

    def forward(self, x, dist, mask):
        x = x.to(torch.int32)
        x = self.embedding(x)
        mem = self.transformer(x, dist, mask).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        return super().forward(mem)


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


if __name__ == "__main__":
    model = SeqNet(depth=3, width=1024, multitask=True).cuda()
    x = torch.rand([8, 285, 1024]).cuda()
    m = torch.ones([8, 285], dtype=bool).cuda()
    for i in range(m.size(0)):
        m[i,:i*2+10] = False
    _, _, pred, _ = model(x, m)
    print(pred.shape) 