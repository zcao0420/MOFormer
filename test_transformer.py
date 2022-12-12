from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerRegressor, Transformer, regressoionHead
from model.utils import *
from datetime import datetime, timedelta
from time import time
from torch.utils.data import dataset, DataLoader

import os
import csv
import yaml
import shutil
import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd
from random import sample
from sklearn import metrics
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_finetune_transformer import MOF_ID_Dataset

def _load_pre_trained_weights(model, config):
    try:
        checkpoints_folder = config['fine_tune_from']
        load_state = torch.load(os.path.join(checkpoints_folder, 'model_transformer_14.pth'),  map_location=config['gpu']) 
        model_state = model.state_dict()
        for name, param in load_state.items():
            if name not in model_state:
                print('NOT loaded:', name)
                continue
            else:
                print('loaded:', name)
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            model_state[name].copy_(param)
        print("Loaded pre-trained model with success.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")
    return model

def get_attention():
    config = yaml.load(open("config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)
    config['gpu'] = 'cpu'
    transformer = Transformer(**config['Transformer'])
    model = _load_pre_trained_weights(transformer, config)
    with open('benchmark_datasets/QMOF/mofid/QMOF_small_mofid.csv') as f:
        reader = csv.reader(f)
        mofdata = [row for row in reader]
    mofdata = np.array(mofdata)
    mofid = mofdata[:, 0]
    maxLen = float('inf')
    shortest = ''
    for i, s in enumerate(mofid):
        if len(s) < maxLen:
            maxLen = len(s)
            shortest = s
    
    tokenizer = MOFTokenizer('tokenizer/vocab_full.txt', model_max_length = 512, padding_side='right')
    token = np.array(tokenizer.encode(shortest, max_length=512, truncation=True,padding='max_length'))
    # model_transformer(torch.from_numpy(token))

    attention = {}
    def hookAttention(name):
        # the hook signature
        def hook(model, input, output):
            attention[name] = output.detach()
        return hook
    h = model.layer-name.register_forward_hook(getActivation(name))
