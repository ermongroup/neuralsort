'''
"Differentiable k-nearest neighbors" layer.

Given a set of M queries and a set of N neighbors,
returns an M x N matrix whose rows sum to k, indicating to what degree
a certain neighbor is one of the k nearest neighbors to the query.
At the limit of tau = 0, each entry is a binary value representing
whether each neighbor is actually one of the k closest to each query.
'''

import torch
from pl import PL
from neuralsort import NeuralSort


class DKNN (torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False, method='deterministic', num_samples=-1):
        super(DKNN, self).__init__()
        self.k = k
        self.soft_sort = NeuralSort(tau=tau, hard=hard)
        self.method = method
        self.num_samples = num_samples

    # query: M x p
    # neighbors: N x p
    #
    # returns:
    def forward(self, query, neighbors, tau=1.0):
        diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
        squared_diffs = diffs ** 2
        l2_norms = squared_diffs.sum(2)
        norms = l2_norms
        scores = -norms

        if self.method == 'deterministic':
            P_hat = self.soft_sort(scores)
            top_k = P_hat[:, :self.k, :].sum(1)
            return top_k
        if self.method == 'stochastic':
            pl_s = PL(scores, tau, hard=False)
            P_hat = pl_s.sample((self.num_samples,))
            top_k = P_hat[:, :, :self.k, :].sum(2)
            return top_k
