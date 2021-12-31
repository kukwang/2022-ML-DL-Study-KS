import numpy as np
import torch
from torch.utils.data import DataLoader
from util.utils import *
from torchvision import datasets
from torchvision import transforms
from util.utils import *
def cifar10(args):
    train_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, transform = get_transform(mode='train'), download=False)
    test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=False, transform = get_transform(mode='test'), download=False)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    return train_dataloader, test_dataloader
    
def cifar100(args):
    train_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, transform = get_transform(mode='train'), download=False)
    test_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=False, transform = get_transform(mode='test'), download=False)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    return train_dataloader, test_dataloader