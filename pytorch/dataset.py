'''
A utility for generating data splits.
'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np


class ClassicDataset(Dataset):

    def __init__(self,
                 x,
                 y,
                 transform):

        self.xy = TensorDataset(x, y)
        self.transform = transform

    def __len__(self):

        return len(self.xy)

    def __getitem__(self, idx):

        x, y = self.xy[idx]
        if self.transform:
            x = self.transform(x)

        return x, y


class DataSplit(object):
    def __init__(self, dataset):
        if dataset == 'mnist':
            trva_real = datasets.MNIST(root='./data-mnist', download=True)
            tr_real_ds, va_real_ds = random_split(trva_real, [55000, 5000])
            xtr_real = trva_real.train_data[tr_real_ds.indices].view(
                -1, 1, 28, 28)
            ytr_real = trva_real.train_labels[tr_real_ds.indices]
            xva_real = trva_real.train_data[va_real_ds.indices].view(
                -1, 1, 28, 28)
            yva_real = trva_real.train_labels[va_real_ds.indices]

            trans = transforms.Compose(
                [transforms.ToPILImage(), transforms.ToTensor()]
            )

            self.train_dataset = ClassicDataset(
                x=xtr_real, y=ytr_real, transform=trans)
            self.valid_dataset = ClassicDataset(
                x=xva_real, y=yva_real, transform=trans)
            self.test_dataset = datasets.MNIST(root='./data-mnist', train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ]))

        elif dataset == 'fashion-mnist':
            trva_real = datasets.FashionMNIST(
                root='./data-fashion-mnist', download=True)
            tr_real_ds, va_real_ds = random_split(trva_real, [55000, 5000])
            xtr_real = trva_real.train_data[tr_real_ds.indices].view(
                -1, 1, 28, 28)
            ytr_real = trva_real.train_labels[tr_real_ds.indices]
            xva_real = trva_real.train_data[va_real_ds.indices].view(
                -1, 1, 28, 28)
            yva_real = trva_real.train_labels[va_real_ds.indices]

            trans = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))]
            )

            self.train_dataset = ClassicDataset(
                x=xtr_real, y=ytr_real, transform=trans)
            self.valid_dataset = ClassicDataset(
                x=xva_real, y=yva_real, transform=trans)
            self.test_dataset = datasets.FashionMNIST(root='./data-fashion-mnist', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.2860,), (.3530,))
            ]))

        elif dataset == 'cifar10':
            trans = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
            )

            trva_real = datasets.CIFAR10(
                root='./data-cifar10', download=True, transform=trans)
            tr_real_ds, va_real_ds = random_split(trva_real, [45000, 5000])

            '''
            tdata = torch.Tensor(trva_real.train_data)
            xtr_real = tdata[tr_real_ds.indices].reshape(
                -1, 3, 32, 32)
            ytr_real = np.array(trva_real.train_labels)[tr_real_ds.indices]
            xva_real = tdata[va_real_ds.indices].reshape(
                -1, 3, 32, 32)
            yva_real = np.array(trva_real.train_labels)[va_real_ds.indices]
            '''

            trans = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
            )

            self.train_dataset = tr_real_ds
            self.valid_dataset = va_real_ds
            self.test_dataset = datasets.CIFAR10(root='./data-cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))

        else:
            raise NotImplementedError()

    def get_train_loader(self, batch_size, **kwargs):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size, num_workers=4, shuffle=True, **kwargs)
        return train_loader

    def get_valid_loader(self, batch_size, **kwargs):
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=batch_size, shuffle=True, **kwargs)
        return valid_loader

    def get_test_loader(self, batch_size, **kwargs):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=batch_size, shuffle=False, **kwargs)
        return test_loader


if __name__ == '__main__':
    from itertools import product

    def get_shapes(loader):
        return [t.shape for t in next(loader.__iter__())]

    dsets = ['mnist', 'fashion-mnist', 'cifar10']
    batch_sizes = [1, 3]
    for d in dsets:
        split = DataSplit(d)
        for b in batch_sizes:
            for loader in split.get_train_loader(1), split.get_valid_loader(2), split.get_test_loader(3):
                print(d, loader, get_shapes(loader))
