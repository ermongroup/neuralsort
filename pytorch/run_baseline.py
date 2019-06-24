import torch
import argparse
import os
import random

import numpy as np

from itertools import islice
from torch import Tensor, FloatTensor, LongTensor
from pl import PL
from utils import one_hot, generate_nothing
from models.preact_resnet import PreActResNet18
from models.easy_net import ConvNet
from dataset import DataSplit
from neuralsort import NeuralSort
from dknn_layer import DKNN

torch.manual_seed(94305)
torch.cuda.manual_seed(94305)
np.random.seed(94305)
random.seed(94305)

parser = argparse.ArgumentParser(
    description="Differentiable k-nearest neighbors.")
parser.add_argument("--k", type=int, metavar="k")
parser.add_argument("--tau", type=float, metavar="tau")
parser.add_argument("--nloglr", type=float, metavar="-log10(beta)")
parser.add_argument("--method", type=str)
parser.add_argument("-resume", action='store_true')
parser.add_argument("--dataset", type=str)

args = parser.parse_args()
dataset = args.dataset
split = DataSplit(dataset)
print(args)

k = args.k
tau = args.tau
NUM_TRAIN_QUERIES = 100
NUM_TEST_QUERIES = 10
NUM_TRAIN_NEIGHBORS = 100
LEARNING_RATE = 10 ** -args.nloglr
NUM_SAMPLES = 5
resume = args.resume
method = args.method

NUM_EPOCHS = 150 if dataset == 'cifar10' else 50
EMBEDDING_SIZE = 500 if dataset == 'mnist' else 512


def experiment_id(dataset, k, tau, nloglr, method):
    return 'baseline-resnet-%s-%s-k%d-t%d-b%d' % (dataset, method, k, tau, nloglr)


e_id = experiment_id(dataset, k, tau * 10, args.nloglr, method)


gpu = torch.device('cuda')

if dataset == 'mnist':
    h_phi = ConvNet().to(gpu)
else:
    h_phi = PreActResNet18(num_channels=3 if dataset ==
                           'cifar10' else 1).to(gpu)

optimizer = torch.optim.SGD(
    h_phi.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

linear_layer = torch.nn.Linear(EMBEDDING_SIZE, 10).to(device=gpu)
ce_loss = torch.nn.CrossEntropyLoss()

batched_train = split.get_train_loader(NUM_TRAIN_QUERIES)


def train(epoch):
    h_phi.train()
    to_average = []
    # train
    for x, y in batched_train:
        optimizer.zero_grad()
        x = x.to(device=gpu)
        y = y.to(device=gpu)
        logits = linear_layer(h_phi(x))
        loss = ce_loss(logits, y)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
    to_average.append((-loss).item())
    print('train', sum(to_average) / len(to_average))


logfile = open('./logs/%s.log' % e_id, 'a' if resume else 'w')

batched_val = split.get_valid_loader(NUM_TEST_QUERIES)
batched_test = split.get_test_loader(NUM_TEST_QUERIES)

best_acc = 0


def test(epoch, val=False):
    h_phi.eval()
    global best_acc
    data = batched_val if val else batched_test

    accs = []

    for x, y in data:
        x = x.to(device=gpu)
        y = y.to(device=gpu)
        logits = linear_layer(h_phi(x))
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean()
        accs.append(acc.item())
    avg_acc = sum(accs) / len(accs)
    print('val' if val else 'test', avg_acc)
    if avg_acc > best_acc and val:
        print('Saving...')
        state = {
            'net': h_phi.state_dict(),
            'acc': avg_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt-%s.t7' % e_id)
        best_acc = avg_acc


for t in range(NUM_EPOCHS):
    print('Beginning epoch %d: ' % t, e_id)
    print('Beginning epoch %d: ' % t, e_id, file=logfile)
    logfile.flush()
    train(t)
    test(t, val=True)


checkpoint = torch.load('./checkpoint/ckpt-%s.t7' % e_id)
h_phi.load_state_dict(checkpoint['net'])
test(-1, val=True)
test(-1, val=False)
logfile.close()
