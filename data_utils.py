import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_mnist_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):

    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = 5000

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    print('train.size={}'.format(len(train_idx)))
    print('valid.size={}'.format(len(valid_idx)))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_mnist_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=False):

    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.MNIST(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    print('test.size={}'.format(len(dataset)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
