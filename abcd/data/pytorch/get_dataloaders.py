"""Get dataloaders, with the option of making them deterministic so that a training state can be 
stored.
"""
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_train_dl(training_data, batch_size, seed=None, at_epoch=0, device='cpu'):
    
    if seed is not None:
        g = torch.Generator(device)
        g.manual_seed(seed + at_epoch)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, 
                                      worker_init_fn=seed_worker, generator=g)
    else:
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    return train_dataloader

def get_eval_dls(datasets, batch_size):
    return {key: DataLoader(ds, batch_size=batch_size, shuffle=False) 
                        for key, ds in datasets.items()}