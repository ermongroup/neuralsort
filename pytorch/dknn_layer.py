import torch
from pl import PL
from sorting_operator import SortingOperator


class DKNN (torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False, method='deterministic', num_samples=-1):
        super(DKNN, self).__init__()
        self.k = k
        self.soft_sort = SortingOperator(tau=tau, hard=hard)
        self.method = method
        self.num_samples = num_samples

    # query: batch_size x p
    # neighbors: 10k x p
    def forward(self, query, neighbors, tau=1.0):
        diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
        squared_diffs = diffs ** 2
        l2_norms = squared_diffs.sum(2)
        norms = l2_norms  # .sqrt() # M x 10k
        scores = -norms

        if self.method == 'deterministic':
            P_hat = self.soft_sort(scores)  # M x 10k x 10k
            top_k = P_hat[:, :self.k, :].sum(1)  # M x 10k
            return top_k
        if self.method == 'stochastic':
            pl_s = PL(scores, tau, hard=False)
            P_hat = pl_s.sample((self.num_samples,))
            top_k = P_hat[:, :, :self.k, :].sum(2)
            return top_k
