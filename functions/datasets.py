import os
import torch
import numpy as np
from torchvision import datasets, transforms


def set_loaders(dataname, batch_size, dogs_data_dir='./data/dogs'):
    """Create test and proxy data loaders.

    The proxy set is a small balanced subset of the test set (10 samples per
    class) used by FedLAW to learn the aggregation parameters.

    Args:
        dataname: one of 'mnist', 'cifar10', 'dogs'.
        batch_size: batch size for both loaders.
        dogs_data_dir: directory containing 'dogs_alltestX.pt' and
                       'dogs_alltesty.pt'. Defaults to './data/dogs'.

    Returns:
        test_loader, proxy_loader
    """
    download = True
    root = '.'

    if dataname == 'mnist':
        num_classes = 10
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,))
        ])
        test_set = datasets.MNIST(root=root, train=False, transform=trans, download=download)

    elif dataname == 'cifar10':
        num_classes = 10
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_set = datasets.CIFAR10(root=root, train=False, transform=trans, download=download)

    elif dataname == 'dogs':
        num_classes = 120
        test_X = torch.load(os.path.join(dogs_data_dir, 'dogs_alltestX.pt'))
        test_y = torch.load(os.path.join(dogs_data_dir, 'dogs_alltesty.pt'))
        test_set = torch.utils.data.TensorDataset(test_X, test_y)

    else:
        raise ValueError(f"Unknown dataset: {dataname}. Choose from 'mnist', 'cifar10', 'dogs'.")

    # Split test set into proxy set (10 samples per class) and evaluation set.
    proxy_size = num_classes * 10
    proxy_and_test = torch.utils.data.random_split(
        dataset=test_set,
        lengths=[proxy_size, len(test_set) - proxy_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=proxy_and_test[1], batch_size=batch_size, shuffle=False
    )
    proxy_loader = torch.utils.data.DataLoader(
        dataset=proxy_and_test[0], batch_size=batch_size, shuffle=False
    )
    return test_loader, proxy_loader


def set_local_data(K, data_dir):
    """Load pre-partitioned local training datasets for K clients.

    Expects files at '{data_dir}/train/0.npz', ..., '{data_dir}/train/{K-1}.npz'.
    Each .npz file must contain a 'data' key whose value is an array with one
    element: a dict with keys 'x' (list of input arrays) and 'y' (list of labels).

    Use prepare_data.py to generate local data in this format from MNIST or CIFAR-10.

    Args:
        K: number of clients.
        data_dir: root directory of the partitioned dataset.

    Returns:
        list of K TensorDataset objects.
    """
    train_datasets = []
    for node in range(K):
        path = f'{data_dir}/train/{node}.npz'
        raw = np.load(path, allow_pickle=True)
        raw = np.atleast_1d(raw['data'])
        inputs = raw[0]['x']
        targets = raw[0]['y']
        tensor_X = torch.stack([torch.from_numpy(np.array(i)) for i in inputs])
        tensor_y = torch.stack([torch.from_numpy(np.array(i)) for i in targets])
        train_datasets.append(torch.utils.data.TensorDataset(tensor_X, tensor_y))
    return train_datasets
