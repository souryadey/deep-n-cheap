# =============================================================================
# Loading data using pytorch
# Ziping Chen, USC
# =============================================================================

import torch
import torchvision
from torchvision import transforms
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Data processing
# =============================================================================
def get_data(data_folder = './', dataset = "mnist", val_split = 1/5, augment = True):
    '''
    Args:
        dataset (string, Dataset from torchvision.datasets): Currently supports MNIST, FMNIST, CIFAR10, CIFAR100
        val_split (float, optional): What fraction of training data to use for validation
            If not 0, val data is taken from end of training set. Eg: For val_split=1/5, last 10k images out of 50k for CIFAR are taken as val
            If 0, train set is complete train set (including val). Test data is returned as val set
            Defaults to 1/5
        augment (bool, optional): If True, do transformations
            Defaults to True
    
    Returns:
        dict: Keys 'train', 'val', 'test'
            Each is a dataset object which can be fed to a data loader
            If val_split is 0, test data is returned as 'val'

    '''
    ## All transforms ##
    if dataset == "mnist":
        dataset = torchvision.datasets.MNIST
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_valtest = transforms.Compose([transforms.ToTensor()])
        
    elif dataset == "fmnist":
        dataset = torchvision.datasets.FashionMNIST
        if not augment: #no transforms
            transform_train = transforms.Compose([transforms.ToTensor()])
            transform_valtest = transforms.Compose([transforms.ToTensor()])
        else: #fancy transforms, without changing dataset size
            mus = (0.2855,)
            sigmas = (0.3528,)
            transform_train = transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mus, sigmas),
                    # transforms.RandomErasing()
                    ])
            transform_valtest = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mus, sigmas)
                    ])
    
    elif dataset == "cifar10" or dataset == "cifar100":
        if dataset == "cifar10":
            dataset = torchvision.datasets.CIFAR10
        else:
            dataset = torchvision.datasets.CIFAR100
        if not augment: #no transforms
            transform_train = transforms.Compose([transforms.ToTensor()])
            transform_valtest = transforms.Compose([transforms.ToTensor()])
        else: #fancy transforms, without changing dataset size
            mus = (0.4914,0.4821,0.4464) if dataset == torchvision.datasets.CIFAR10 else (0.507,0.4867,0.441)
            sigmas = (0.2471,0.2436,0.2616) if dataset == torchvision.datasets.CIFAR10 else (0.2674,0.2564,0.276)
            transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mus, sigmas),
                    # transforms.RandomErasing()
                    ])
            transform_valtest = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mus, sigmas)
                    ])
        
    ## Create actual datasets ##
    if val_split > 1e-8: #val data exists
        train_full = dataset(root=data_folder, train=True, download=True, transform=transform_train)
        val_full = dataset(root=data_folder, train=True, download=True, transform=transform_valtest)
        split = int((1-val_split)*len(train_full))
        
        train = torch.utils.data.Subset(train_full, range(split))
        val = torch.utils.data.Subset(val_full, range(split, len(val_full)))
        test = dataset(root=data_folder, train=False, download=True, transform=transform_valtest)
        return {'train':train, 'val':val, 'test':test}
    
    else: #test is val data, actual val data is combined with train
        train = dataset(root=data_folder, train=True, download=True, transform=transform_train)
        test = dataset(root=data_folder, train=False, download=True, transform=transform_valtest)
        return {'train':train, 'val':test, 'test':test} #doesn't matter that we return test as 'test', it won't be used in that sense

def get_data_npz(data_folder = './', dataset = 'fmnist.npz', val_split = 1/5, problem_type = 'classification'):
    '''
    Args:
        data_folder : Location of dataset
        dataset (string): <dataset name>.npz, must have 4 keys -- xtr, ytr, xte, yte
            xtr: (num_trainval_samples, num_features...)
            ytr: (num_trainval_samples,)
            xte: (num_test_samples, num_features...)
            yte: (num_test_samples,)
        val_split (float, optional): What fraction of training data to use for validation
            If not 0, val data is taken from end of training set. Eg: For val_split=1/6, last 10k images out of 60k for MNIST are taken as val
            If 0, train set is complete train set (including val). Test data is returned as val set
            Defaults to 1/5
        problem_type (string, required): Task/problem type. Required for the dtype of the labels.

    Returns:
        xtr (torch tensor): Shape: (num_train_samples, num_features...)
        ytr (torch tensor): Shape: (num_train_samples,)
        xva (torch tensor): Shape: (num_val_samples, num_features...). This is xte if val_split = 0
        yva (torch tensor): Shape: (num_val_samples,). This is yte if val_split = 0
        xte (torch tensor): Shape: (num_test_samples, num_features...)
        yte (torch tensor): Shape: (num_test_samples,)

    '''
    loaded = np.load(os.path.join(data_folder, dataset))
    xtr = loaded['xtr']
    ytr = loaded['ytr']
    xte = loaded['xte']
    yte = loaded['yte'] 
    
    ## Convert to tensors on device ##
    xtr = torch.as_tensor(xtr, dtype=torch.float, device=device)
    xte = torch.as_tensor(xte, dtype=torch.float, device=device)
    if problem_type == 'classification':
        ytr = torch.as_tensor(ytr, dtype=torch.long, device=device)
        yte = torch.as_tensor(yte, dtype=torch.long, device=device)
    elif problem_type == 'regression':
        ytr = torch.as_tensor(ytr, dtype=torch.float, device=device)
        yte = torch.as_tensor(yte, dtype=torch.float, device=device)

    if abs(val_split) < 1e-8:
        # val_spilt is 0.0
        return xtr,ytr, xte,yte, xte,yte
    else:
        split = int((1-val_split)*len(xtr))
        xva = xtr[split:]
        yva = ytr[split:]
        xtr = xtr[:split]
        ytr = ytr[:split]
        xva = torch.as_tensor(xva, dtype=torch.float, device=device)
        if problem_type == 'classification':
            yva = torch.as_tensor(yva, dtype=torch.long, device=device)
        elif problem_type == 'regression':
            yva = torch.as_tensor(yva, dtype=torch.float, device=device)
        return xtr,ytr, xva,yva, xte,yte
        
