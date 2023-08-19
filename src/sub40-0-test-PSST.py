#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import sys,os
import time
import numpy as np
import torch
import torch as pt
import argparse
import warnings

from torch import nn, optim
from torch.autograd import Variable
from scipy.spatial.distance import pdist, squareform

from model import *

from utils import readpickle, savepickle
import torch.nn.functional as F
import time
from torch.nn import DataParallel


valset = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/data_val_sub40.pkl')
testset = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/data_test_sub40.pkl')
print(len(valset))
print(len(testset))


modelfn = '/cmach-data/liuyongchang/protein_sfc/PSST/checkpoints/model_88_0.8697.pth'
model = PSST(depth=3, width=1024, multitask=True)
model = DataParallel(model)
model = model.cuda()
model.load_state_dict(pt.load(modelfn, map_location='cpu'))
model.eval()


n_epoch = 100
LR = 1e-4
BS = 1
optimizer = optim.Adam(model.parameters(), lr=LR)


val_correct_cc = 0
val_total_cc = 0
val_correct_ff = 0
val_total_ff = 0
val_correct = 0
val_total = 0
sid_list_val = []
label_list_val = []
length_list_val = []
flag_list_val = []
predictloader_val = dataloader_test(valset, BS)
for x, d, m, pclass, fold, label, sids, lengths in predictloader_val:
    x = Variable(x).cuda()
    d = Variable(d).cuda()
    m = Variable(m).cuda()
    pclass = pclass.squeeze(-1).cuda()
    fold = fold.squeeze(-1).cuda()
    label = label.squeeze(-1).cuda()

    model = model.eval()
    cc, ff, pred, _ = model(x, d, m)

    _, predicted_cc = torch.max(cc.data, 1)
    val_total_cc += pclass.size(0)
    val_correct_cc += (predicted_cc == pclass).sum().item()

    _, predicted_ff = torch.max(ff.data, 1)
    val_total_ff += fold.size(0)
    val_correct_ff += (predicted_ff == fold).sum().item()

    _, predicted = torch.max(pred.data, 1)
    val_total += label.size(0)
    val_correct += (predicted == label).sum().item()

    sid_list_val.append(sids[0])
    label_list_val.append(label[0])
    length_list_val.append(lengths[0])
    flag_list_val.append((predicted == label).sum().item())


val_acc_cc = val_correct_cc / val_total_cc
print("val accuracy class: {}".format(round(val_acc_cc, 4)))
val_acc_ff = val_correct_ff / val_total_ff
print("val accuracy fold: {}".format(round(val_acc_ff, 4)))
val_acc = val_correct / val_total
print("val accuracy superfamily: {}".format(round(val_acc, 4)))


test_correct_cc = 0
test_total_cc = 0
test_correct_ff = 0
test_total_ff = 0
test_correct = 0
test_total = 0
sid_list_test = []
label_list_test = []
length_list_test = []
flag_list_test = []
predictloader_test = dataloader_test(testset, BS)
for x, d, m, pclass, fold, label, sids, lengths in predictloader_test:
    x = Variable(x).cuda()
    d = Variable(d).cuda()
    m = Variable(m).cuda()
    pclass = pclass.squeeze(-1).cuda()
    fold = fold.squeeze(-1).cuda()
    label = label.squeeze(-1).cuda()

    model = model.eval()
    cc, ff, pred, _ = model(x, d, m)

    _, predicted_cc = torch.max(cc.data, 1)
    test_total_cc += pclass.size(0)
    test_correct_cc += (predicted_cc == pclass).sum().item()

    _, predicted_ff = torch.max(ff.data, 1)
    test_total_ff += fold.size(0)
    test_correct_ff += (predicted_ff == fold).sum().item()

    _, predicted = torch.max(pred.data, 1)
    test_total += label.size(0)
    test_correct += (predicted == label).sum().item()

    sid_list_test.append(sids[0])
    label_list_test.append(label[0])
    length_list_test.append(lengths[0])
    flag_list_test.append((predicted == label).sum().item())

test_acc_cc = test_correct_cc / test_total_cc
print("test accuracy class: {}".format(round(test_acc_cc, 4)))
test_acc_ff = test_correct_ff / test_total_ff
print("test accuracy fold: {}".format(round(test_acc_ff, 4)))
test_acc = test_correct / test_total
print("test accuracy superfamily: {}".format(round(test_acc, 4)))


result = [sid_list_val, length_list_val, flag_list_val, sid_list_test, length_list_test, flag_list_test]
savepickle(result, './save/PSST_sub40result.pkl')

# CUDA_VISIBLE_DEVICES=0 python sub40-0-test-PSST.py