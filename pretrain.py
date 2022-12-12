import pandas as pd
import logging
import numpy as np
import torch
import math, os, shutil
import functools
import matplotlib.pyplot as plt
from typing import Tuple
from torch import nn, Tensor
from torch._C import TensorType
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data import dataset, DataLoader
# from transformers.utils.dummy_vision_objects import ImageFeatureExtractionMixin
from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import Transformer
from model.utils import *
from model.dataset import MOF_pretrain_Dataset
from datetime import datetime, timedelta
from time import time
from model.mlm_pytorch_new import MLM
from tqdm import tqdm

def plot_loss(CE_loss, filename):
    plt.figure(figsize = [4,3], dpi = 150)
    plt.plot(CELoss, label = 'Training')
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy loss')
    plt.legend(frameon = False)

    plt.savefig(filename, bbox_inches = 'tight')


if __name__ == '__main__':
    model_param = {
        'd_model': 512,
        'nhead': 8,
        'd_hid': 512,
        'nlayers': 6,
        'dropout': 0.1
    }
    start = time()
    now = datetime.now()

    folderName = '_'.join([
        str(model_param['d_model']), str(model_param['nhead']), 
        str(model_param['d_hid']), str(model_param['nlayers'])
    ])
    newFolder = os.path.join('training_results/pretraining', folderName)
    if not os.path.exists(newFolder):
        os.mkdir(newFolder)
    ################################
    torch.cuda.empty_cache()
    
    mofdata = np.load('data/large_pretrain_512.npy', allow_pickle=True)
    vocab_path = 'tokenizer/vocab_full.txt'
    tokenizer = MOFTokenizer(vocab_path, model_max_length = 512, padding_side='right')

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(device)
    if device == torch.device('cpu'):
        num_workers = 0
    else:
        num_workers = 4

    train_dataset = MOF_pretrain_Dataset(mofdata, tokenizer, use_ratio=1)

    train_loader = DataLoader(
        train_dataset, batch_size=32, num_workers=num_workers, drop_last=False, 
        shuffle=True, pin_memory = False
    )
    print('Total number of MOFid in pretraining set: %d'%len(train_dataset))
    ######## Models #############
    model = Transformer(
        ntoken=4021, d_model=model_param['d_model'], nhead=model_param['nhead'], 
        d_hid=model_param['d_hid'], nlayers=model_param['nlayers'], dropout=model_param['dropout']
    )

    mlp = nn.Sequential(
        nn.LayerNorm(512),
        nn.Identity(),
        nn.Linear(512, 4021) 
    )

    trainer = MLM(
        model,
        mlp,
        mask_token_id = 14,          # the token id reserved for masking
        pad_token_id = 0,           # the token id for padding
        mask_prob = 0.15,           # masking probability for masked language modeling
        replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
        mask_ignore_token_ids = [12, 13]  # other tokens to exclude from masking, include the [cls] and [sep] here
    ).to(device)

    n_epoch = 100
    # optimizer = optim.AdamW(trainer.parameters(), lr=5e-5, weight_decay=1e-4)
    optimizer = optim.AdamW(trainer.parameters(), lr=5e-5)


    CELoss = []
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0
        trainmse = []
        trainmae = []
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            data = data.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(data)

            loss = trainer(data)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(train_loader)
        print('Epoch: %d | Loss: %.3f'%(epoch+1, epoch_loss))
        CELoss.append(epoch_loss)
        if epoch%5 == 0:
            img_name = os.path.join('training_results/pretraining', folderName, 'loss.png')
            plot_loss(CELoss, img_name)
            torch.save(model.state_dict(), os.path.join('training_results/pretraining', folderName, 'pretrained-model.pt'))
    # print(CELoss)
    ########## Time Block ##############
    elapsed = (time() - start)
    print('Training time: %s'%str(timedelta(seconds=elapsed)))
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%H_%M")
    print("Finished training at:", dt_string)
    ####################################
    plt.figure(figsize = [4,3], dpi = 150)
    plt.plot(CELoss, label = 'Training')
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy loss')
    plt.legend(frameon = False)

    plt.savefig(os.path.join('training_results/pretraining', folderName, 'loss.png'), bbox_inches = 'tight')

    np.save(
        os.path.join('training_results/pretraining', folderName, 'CE_loss.npy'), 
        CELoss
    )
    shutil.copy2('pretrain.py', os.path.join('training_results/pretraining', folderName))
    torch.save(model.state_dict(), os.path.join('training_results/pretraining', folderName, 'pretrained-model.pt'))
    print('result_saved')