import torch

# labels is a 1-dimensional tensor


def one_hot(labels, l=10):
    n = labels.shape[0]
    labels = labels.unsqueeze(-1)
    oh = torch.zeros(n, l, device='cuda').scatter_(1, labels, 1)
    return oh


generate_nothing = iter(int, 1)
