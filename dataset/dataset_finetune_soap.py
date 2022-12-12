from __future__ import print_function, division

import csv
import functools
import json
from logging import root
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

# from augmentation import RotationTransformation, PerturbStructureTransformation, SwapAxesTransformation


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64,random_seed = 2, val_ratio=0.1, test_ratio=0.1, 
                              return_test=False, num_workers=1, pin_memory=False, 
                              **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    # if train_ratio is None:
    #     assert val_ratio + test_ratio < 1
    #     train_ratio = 1 - val_ratio - test_ratio
    #     # print('[Warning] train_ratio is None, using all training data.')
    # else:
    #     assert train_ratio + val_ratio + test_ratio <= 1
    train_ratio = 1 - val_ratio - test_ratio
    indices = list(range(total_size))
    print("The random seed is: ", random_seed)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    print('Train size: {}, Validation size: {}, Test size: {}'.format(
        train_size, valid_size, test_size
    ))
    
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
            torch.stack(batch_target, dim=0),\
            batch_cif_ids


class SOAPData(Dataset):
    """
    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self,task, root_dir, label_dir,
                 random_seed=123):
        self.root_dir = root_dir
        self.task = task
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = label_dir
        assert os.path.exists(id_prop_file), 'id_prop_file does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        self.id_prop_data = self.id_prop_data
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
     

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]


        soap = np.load(os.path.join(self.root_dir, cif_id+'_soap.npz'))['soap']

        if self.task == 'regression':
            target = torch.Tensor([float(target)])
        else:
            if target == 'True':
                label = 0
            elif target == 'False':
                label = 1
            target = torch.Tensor([float(label)])
        
        return soap, target, cif_id


class SOAP_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, root_dir):
        self.data = data
        self.root_dir = root_dir
    #     self.data = data[:int(len(data)*use_ratio)]
        self.cifid = self.data[:, 0].astype(str)
        self.label = self.data[:, 1].astype(float)

    def __len__(self):
        return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
        # Load data and get label
        soap = np.load(
            os.path.join(self.root_dir, self.cifid[index]+'_soap.npz'), 
            allow_pickle = True
        )['soap']
        X = torch.from_numpy(soap).float()
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1,1)
        return X, y.float()