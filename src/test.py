
import sys,os
import time
import numpy as np
import torch
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


testset = readpickle('../save/data_test.pkl')

modelfn = '../checkpoints/model_88_0.8697.pth'
model = PSST(depth=3, width=1024, multitask=True)
model = DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load(modelfn, map_location='cpu'))
model.eval()
BS = 1

test_correct = 0
test_total = 0
dataloader_test = dataloader(testset, BS)
for x, d, m, pclass, fold, label in dataloader_test:
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