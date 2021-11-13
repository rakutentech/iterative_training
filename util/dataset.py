from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random

import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset


def load_mnist(args, data_folder='./data', num_workers=2):
    """MNIST
    http://yann.lecun.com/exdb/mnist/
    60000 train / 10000 test
    """
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_folder, train=True,
                       download=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_folder, train=False, transform=transform),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)
    return train_loader, test_loader


def load_mnist_with_validation(args, data_folder='./data', num_workers=2, validation_size=5000):
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(
        data_folder, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_folder, train=False, transform=transform)
    validation_split = len(train_data) - validation_size
    train_dataset = Subset(train_data, range(0, validation_split))
    validation_dataset = Subset(train_data, range(
        validation_split, len(train_data)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)
    return train_loader, validation_loader, test_loader


def load_small_mnist(args, data_folder='./data', num_workers=2, train_size=10000, test_size=2000):
    assert(train_size <= 60000)
    assert(test_size <= 10000)
    assert(train_size + test_size <= 60000)
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(
        data_folder, train=True, download=True, transform=transform)
    train_dataset = Subset(train_data, range(0, train_size))
    validation_dataset = Subset(
        train_data, range(train_size, train_size+test_size))
    test_data = datasets.MNIST(data_folder, train=False, transform=transform)
    test_dataset = Subset(test_data, range(0, test_size))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)
    return train_loader, validation_loader, test_loader


def load_cifar10(args, data_folder='./data', num_workers=2):
    """CIFAR-10
    https://www.cs.toronto.edu/~kriz/cifar.html
    50000 train / 10000 test
    """
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_data = datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_data = datasets.CIFAR10(
        root=data_folder, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def get_cifar10_transforms(use_data_augmentation):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ])
    if use_data_augmentation:
        return transform_train, transform_test
    else:
        return transform_test, transform_test


def load_cifar10_with_validation(
    args,
    data_folder='./data',
    num_workers=2,
    validation_size=5000
):
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform_train, transform_test = get_cifar10_transforms(
        args.use_data_augmentation)

    train_data = datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train)
    assert (validation_size < len(train_data))
    sets = [len(train_data)-validation_size, validation_size]
    print("Dataset splits {}".format(sets))
    splits = torch.utils.data.random_split(train_data, sets)
    train_loader = torch.utils.data.DataLoader(
        splits[0], batch_size=args.batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        splits[1], batch_size=args.batch_size, shuffle=True, **kwargs)

    test_data = datasets.CIFAR10(
        root=data_folder, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return train_loader, validation_loader, test_loader


def load_cifar10_with_preselected_validation(
    batch_size,
    test_batch_size,
    use_data_augmentation,
    index_file='cifar10_split.json',
    data_folder='./data',
    num_workers=2,
):
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform_train, transform_test = get_cifar10_transforms(use_data_augmentation)

    train_indices, val_indices = load_train_val_indices(index_file)
    train_data = datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train)
    assert (len(train_data) == len(train_indices) + len(val_indices))

    train_dataset = Subset(train_data, train_indices)
    validation_dataset = Subset(train_data, val_indices)
    print("Dataset splits {}/{}".format(len(train_indices), len(val_indices)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    test_data = datasets.CIFAR10(
        root=data_folder, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, validation_loader, test_loader


def load_cifar10_with_preselected_validation_fix_validation_transform(
    batch_size,
    test_batch_size,
    use_data_augmentation,
    index_file='cifar10_split.json',
    data_folder='./data',
    num_workers=2,
):
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform_train, transform_test = get_cifar10_transforms(use_data_augmentation)

    train_indices, val_indices = load_train_val_indices(index_file)
    train_data = datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train)
    assert (len(train_data) == len(train_indices) + len(val_indices))
    train_dataset = Subset(train_data, train_indices)
    validation_data = datasets.CIFAR10(
        root=data_folder, train=True, download=False, transform=transform_test)
    validation_dataset = Subset(validation_data, val_indices)
    test_data = datasets.CIFAR10(
        root=data_folder, train=False, transform=transform_test)
    print("Dataset splits {}/{}/{}".format(len(train_indices), len(val_indices), len(test_data)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, validation_loader, test_loader



def load_small_cifar10_with_validation(
    args,
    data_folder='./data',
    num_workers=2,
    train_size=10000,
    validation_size=2000,
    test_size=2000
):
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform_train, transform_test = get_cifar10_transforms(
        args.use_data_augmentation)

    train_data = datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train)
    assert (train_size+validation_size < len(train_data))
    rest = len(train_data) - train_size - validation_size
    sets = [train_size, validation_size, rest]
    print("Dataset splits {}".format(sets))
    splits = torch.utils.data.random_split(train_data, sets)
    train_loader = torch.utils.data.DataLoader(
        splits[0], batch_size=args.batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        splits[1], batch_size=args.batch_size, shuffle=True, **kwargs)

    test_data = datasets.CIFAR10(
        root=data_folder, train=False, transform=transform_test)
    assert (test_size < len(test_data))
    sets = [test_size, len(test_data) - test_size]
    print("Dataset splits {}".format(sets))
    splits = torch.utils.data.random_split(test_data, sets)
    test_loader = torch.utils.data.DataLoader(
        splits[0], batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return train_loader, validation_loader, test_loader


def generate_train_val_indices(input_size, val_size, file_name):
    """Generate random indices for training and validation set

    Parameters
    ==========
    input_size : int
        Dataset size.
    val_size : int
        Validation size.
    file_name : str
        Output json file name.

    Returns
    =======
    train_indices : list(int)
    val_indices : list(int)
    """
    assert (val_size < input_size)
    indices = [x for x in range(0, input_size)]
    random.shuffle(indices)
    train_indices = indices[val_size:]
    validation_indices = indices[:val_size]
    print("train size {} val size {}".format(
        len(train_indices), len(validation_indices)))

    # Write indices to a json file
    with open(file_name, 'w') as fp:
        data = {'train': train_indices, 'validation': validation_indices}
        json.dump(data, fp)
        print("Saved to {}...".format(file_name))

    return train_indices, validation_indices


def load_train_val_indices(file_name):
    """Generate random indices for training and validation set

    Parameters
    ==========
    file_name : str
        Output json file name.

    Returns
    =======
    train_indices : list(int)
    val_indices : list(int)
    """

    with open(file_name, 'r') as fp:
        print("Loading from {}...".format(file_name))
        data = json.load(fp)
        return data['train'], data['validation']
