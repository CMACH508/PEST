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

from utils import readpickle
import torch.nn.functional as F
import time
from torch.nn import DataParallel


trainset = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/data_train_sub40.pkl')
valset = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/data_val_sub40.pkl')
testset = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/data_test_sub40.pkl')
print(len(trainset))
print(len(valset))
print(len(testset))

checkpoint_path = '/cmach-data/liuyongchang/protein_sfc/PSST/checkpoints'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

model = PSST(depth=3, width=1024, multitask=True)
model = DataParallel(model)
model = model.cuda()

n_epoch = 100
LR = 1e-4
BS = 32
optimizer = optim.Adam(model.parameters(), lr=LR)

best_acc = 0

for epoch in range(n_epoch):
    print("Epoch: {}".format(epoch))
    s = time.time()

    train_correct = 0
    train_total = 0
    predictloader_train = dataloader(trainset, BS)

    for x, d, m, pclass, fold, label in predictloader_train:
        x = Variable(x).cuda()
        d = Variable(d).cuda()
        m = Variable(m).cuda()
        pclass = pclass.squeeze().cuda()
        fold = fold.squeeze().cuda()
        label = label.squeeze().cuda()

        optimizer.zero_grad()
        model = model.train()
        class_pred, fold_pred, pred, embed = model(x, d, m)

        _, predicted = torch.max(pred.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()

        loss_class = F.cross_entropy(class_pred, pclass)
        loss_fold = F.cross_entropy(fold_pred, fold)
        loss_superf = F.cross_entropy(pred, label)

        squared = torch.pow(embed, 2)
        me = torch.mean(squared, dim=1)
        rms = torch.sqrt(me)
        rms_loss = torch.mean(rms)

        loss = loss_class + loss_fold + loss_superf + 0.1 * rms_loss
        loss.backward()
        optimizer.step()

    train_acc = train_correct / train_total
    print("train accuracy: {}".format(round(train_acc, 4)))

    val_correct = 0
    val_total = 0
    predictloader_val = dataloader(valset, BS)

    for x, d, m, pclass, fold, label in predictloader_val:
        x = Variable(x).cuda()
        d = Variable(d).cuda()
        m = Variable(m).cuda()
        label = label.squeeze(-1).cuda()

        model = model.eval()
        _, _, pred, _ = model(x, d, m)

        _, predicted = torch.max(pred.data, 1)
        val_total += label.size(0)
        val_correct += (predicted == label).sum().item()

    val_acc = val_correct / val_total
    print("val accuracy: {}".format(round(val_acc, 4)))

    e = time.time()
    print("Elapsed time: {}".format(e - s))

    if val_acc > best_acc:
        print('Saving..')
        torch.save(model.state_dict(), '{}/model_{}_{}.pth'.format(checkpoint_path, epoch, round(val_acc, 4)))
        best_acc = val_acc

    print("best val accuracy: {}".format(best_acc))


    test_correct = 0
    test_total = 0
    predictloader_test = dataloader(testset, BS)
    for x, d, m, pclass, fold, label in predictloader_test:
        x = Variable(x).cuda()
        d = Variable(d).cuda()
        m = Variable(m).cuda()
        label = label.squeeze(-1).cuda()

        model = model.eval()
        _, _, pred, _ = model(x, d, m)

        _, predicted = torch.max(pred.data, 1)
        test_total += label.size(0)
        test_correct += (predicted == label).sum().item()

    test_acc = test_correct / test_total
    print("test accuracy: {}".format(round(test_acc, 4)))


# CUDA_VISIBLE_DEVICES=2,3 nohup python -u sub40-0-train-PSST.py > sub40-0-train-PSST-out.txt 2>&1 &
