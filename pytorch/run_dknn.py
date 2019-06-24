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
parser.add_argument("--k", type=int, metavar="k", required=True)
parser.add_argument("--tau", type=float, metavar="tau", default=16.)
parser.add_argument("--nloglr", type=float, metavar="-log10(beta)", default=3.)
parser.add_argument("--method", type=str, default="deterministic")
parser.add_argument("-resume", action='store_true')
parser.add_argument("--dataset", type=str, required=True)

parser.add_argument("--num_train_queries", type=int, default=100)
# no effect on training, but massive effect on memory usage
parser.add_argument("--num_test_queries", type=int, default=10)
parser.add_argument("--num_train_neighbors", type=int, default=100)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--num_epochs", type=int, default=200)

args = parser.parse_args()
dataset = args.dataset
split = DataSplit(dataset)
print(args)

k = args.k
tau = args.tau
NUM_TRAIN_QUERIES = args.num_train_queries
NUM_TEST_QUERIES = args.num_test_queries
NUM_TRAIN_NEIGHBORS = args.num_train_neighbors
LEARNING_RATE = 10 ** -args.nloglr
NUM_SAMPLES = args.num_samples
resume = args.resume
method = args.method
NUM_EPOCHS = args.num_epochs
EMBEDDING_SIZE = 500 if dataset == 'mnist' else 512


def experiment_id(dataset, k, tau, nloglr, method):
    return 'dknn-resnet-%s-%s-k%d-t%d-b%d' % (dataset, method, k, tau * 100, nloglr)


e_id = experiment_id(dataset, k, tau, args.nloglr, method)


dknn_layer = DKNN(k, tau, method=method, num_samples=NUM_SAMPLES)


def dknn_loss(query, neighbors, query_label, neighbor_labels, method=method):
    # query: batch_size x p
    # neighbors: 10k x p
    # query_labels: batch_size x [10] one-hot
    # neighbor_labels: n x [10] one-hot
    if method == 'deterministic':
        top_k_ness = dknn_layer(query, neighbors)
        correct = (query_label.unsqueeze(1) *
                   neighbor_labels.unsqueeze(0)).sum(-1)
        correct_in_top_k = (correct * top_k_ness).sum(-1)
        loss = -correct_in_top_k
        return loss
    elif method == 'stochastic':
        top_k_ness = dknn_layer(query, neighbors)
        correct = (query_label.unsqueeze(1) *
                   neighbor_labels.unsqueeze(0)).sum(-1)
        correct_in_top_k = (correct.unsqueeze(0) * top_k_ness).sum(-1)
        loss = -correct_in_top_k
        return loss
    else:
        raise ValueError(method)


gpu = torch.device('cuda')

if dataset == 'mnist':
    h_phi = ConvNet().to(gpu)
else:
    h_phi = PreActResNet18(num_channels=3 if dataset ==
                           'cifar10' else 1).to(gpu)

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt-%s.t7' % e_id)
    h_phi.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    best_acc = 0
    start_epoch = 0


optimizer = torch.optim.SGD(
    h_phi.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

unit_test_linear_layer = torch.nn.Linear(EMBEDDING_SIZE, 10).to(device=gpu)
unit_test_ce_loss = torch.nn.CrossEntropyLoss()

ema_factor = .999
ema_num = 0


batched_query_train = split.get_train_loader(NUM_TRAIN_QUERIES)
batched_neighbor_train = split.get_train_loader(NUM_TRAIN_NEIGHBORS)


def train(epoch):
    h_phi.train()
    to_average = []
    # train
    for query, candidates in zip(batched_query_train, batched_neighbor_train):
        optimizer.zero_grad()
        cand_x, cand_y = candidates
        query_x, query_y = query

        cand_x = cand_x.to(device=gpu)
        cand_y = cand_y.to(device=gpu)
        query_x = query_x.to(device=gpu)
        query_y = query_y.to(device=gpu)

        neighbor_e = h_phi(cand_x).reshape(NUM_TRAIN_NEIGHBORS, EMBEDDING_SIZE)
        query_e = h_phi(query_x).reshape(NUM_TRAIN_QUERIES, EMBEDDING_SIZE)

        neighbor_y_oh = one_hot(cand_y).reshape(NUM_TRAIN_NEIGHBORS, 10)
        query_y_oh = one_hot(query_y).reshape(NUM_TRAIN_QUERIES, 10)

        losses = dknn_loss(query_e, neighbor_e, query_y_oh, neighbor_y_oh)
        loss = losses.mean()
        loss.backward()
        optimizer.step()
        to_average.append((-loss).item() / k)

    print('Avg. train correctness of top k:',
          sum(to_average) / len(to_average))
    print('Avg. train correctness of top k:', sum(
        to_average) / len(to_average), file=logfile)
    logfile.flush()


def majority(lst):
    return max(set(lst), key=lst.count)


def new_predict(query, neighbors, neighbor_labels):
    '''
    query: p
    neighbors: n x p
    neighbor_labels: n (int)
    '''
    diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
    squared_diffs = diffs ** 2
    norms = squared_diffs.sum(-1)
    indices = torch.argsort(norms, dim=-1)
    labels = neighbor_labels.take(indices[:, :k])
    prediction = [majority(l.tolist()) for l in labels]
    return torch.Tensor(prediction).to(device=gpu).long()


def acc(query, neighbors, query_label, neighbor_labels):
    prediction = new_predict(query, neighbors, neighbor_labels)
    return (prediction == query_label).float().cpu().numpy()


logfile = open('./logs/%s.log' % e_id, 'a' if resume else 'w')

batched_query_val = split.get_valid_loader(NUM_TEST_QUERIES)
batched_query_test = split.get_test_loader(NUM_TEST_QUERIES)


def test(epoch, val=False):
    h_phi.eval()
    global best_acc
    to_average = []
    with torch.no_grad():
        embeddings = []
        labels = []
        for neighbor_x, neighbor_y in batched_neighbor_train:
            neighbor_x = neighbor_x.to(device=gpu)
            neighbor_y = neighbor_y.to(device=gpu)
            embeddings.append(h_phi(neighbor_x))
            labels.append(neighbor_y)
        neighbors_e = torch.stack(embeddings).reshape(-1, EMBEDDING_SIZE)
        labels = torch.stack(labels).reshape(-1)

        estimate_batches = []
        results = []
        for queries in batched_query_val if val else batched_query_test:
            query_x, query_y = queries
            query_x = query_x.to(device=gpu)
            query_y = query_y.to(device=gpu)
            query_e = h_phi(query_x)
            results.append(acc(query_e, neighbors_e, query_y, labels))
        total_acc = np.mean(np.array(results))
        total_acc_batch = np.mean(results[0])

    split = 'val' if val else 'test'
    print('Avg. %s acc:' % split, total_acc)
    print('Avg. %s acc:' % split, total_acc, file=logfile)
    if total_acc > best_acc and val:
        print('Saving...')
        state = {
            'net': h_phi.state_dict(),
            'acc': total_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt-%s.t7' % e_id)
        best_acc = total_acc


for t in range(start_epoch, NUM_EPOCHS):
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
