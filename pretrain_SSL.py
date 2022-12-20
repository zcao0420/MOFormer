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
#from model.dataset import MOF_pretrain_Dataset
from dataset.dataset_multiview import CIFData,collate_pool
from datetime import datetime, timedelta
from time import time
from model.mlm_pytorch_new import MLM
from loss.barlow_twins import BarlowTwinsLoss

from tqdm import tqdm


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_multiview.yaml', os.path.join(model_checkpoints_folder, 'config_multiview.yaml'))

class Multiview(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time
        log_dir = os.path.join('runs_mutltiview', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        self.criterion = BarlowTwinsLoss(self.device, config['batch_size'], **config['loss'])
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length = 512, padding_side='right')
        self.dataset = CIFData(**self.config['graph_dataset'], tokenizer = self.tokenizer)

        collate_fn = collate_pool
        self.train_loader, self.valid_loader = get_train_val_test_loader(
            dataset=self.dataset,
            collate_fn=collate_fn,
            pin_memory=self.config['cuda'],
            batch_size=self.config['batch_size'], 
            **self.config['dataloader']
        )


    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model_t,model_g, model_mlp, data_i, data_j):
        # get the representations and the projections
        _, zis = model_t(data_i)  # [N,C]
        # get the representations and the projections
        _, zjs = model_g(data_j)  # [N,C]

        # normalize projection feature vectors
        # zis = F.normalize(zis, dim=1)
        # zjs = F.normalize(zjs, dim=1)

        loss = self.criterion(zis, zjs)
        return loss

    def train(self):

        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]

        model_t = Transformer(**self.config["transformer"]).to(self.device)

        model_g = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    **self.config['model'])

        # mlp = nn.Sequential(
        #     nn.LayerNorm(512),
        #     nn.Identity(),
        #     nn.Linear(512, 4021) 
        #     )
        
        # trainer = MLM(model,mlp,**self.config["MLM"]).to(device)
        model_t, model_g = self._load_pre_trained_weights(model_t,model_g)
        # model_g = self._load_pre_trained_weights(model_g)


        optimizer = torch.optim.Adam((model_t.parameters(),model_g.parameters()), self.config['init_lr'], weight_decay=eval(self.config['weight_decay']))
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, (input_1, input_2, _) in enumerate(self.train_loader):
                if self.config['cuda']:
                    input_graph = (Variable(input_1[0].to(self.device, non_blocking=True)),
                                Variable(input_1[1].to(self.device, non_blocking=True)),
                                input_1[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_1[3]])
                    input_transformer = input_2.to(self.device, non_blocking = True)

                else:
                    input_graph = (Variable(input_1[0]),
                                Variable(input_1[1]),
                                input_1[2],
                                input_1[3])
                    input_transformer = input_2
                
                loss = self._step(model_t, model_g, input_transformer,input_graph)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            torch.cuda.empty_cache()
            #print("1st",os.system('free -h'))  
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, self.valid_loader)
                print('Validation', valid_loss)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if epoch_counter > 0 and epoch_counter % 1 == 0:
                torch.save(model_t.state_dict(), os.path.join(model_checkpoints_folder, 'model_transformer_{}.pth'.format(epoch_counter)))
                torch.save(model_g.state_dict(), os.path.join(model_checkpoints_folder, 'model_graph_{}.pth'.format(epoch_counter)))
            
            # warmup for the first 5 epochs
            if epoch_counter >= 5:
                scheduler.step()
    
    def _load_pre_trained_weights(self, model_t,model_g):
        #print("Here")
        try:
            checkpoints_folder = os.path.join('./runs_multiview', self.config['fine_tune_from'], 'checkpoints')
            state_dict_t = torch.load(os.path.join(checkpoints_folder, 'model_transformer_0.pth'))
            model_t.load_state_dict(state_dict_t)

            state_dict_g = torch.load(os.path.join(checkpoints_folder, 'model_graph_0.pth'))
            model_g.load_state_dict(state_dict_g)

            print("Loaded pre-trained model with success.")
            #print("loaded")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model_t, model_g

    def _validate(self, model_t, model_g, valid_loader):
        # validation steps
        #print("2st",os.system('free -h'))
        with torch.no_grad():
            model.eval()

            loss_total = 0.0
            total_num = 0
            for input_1, input_2, batch_cif_ids in valid_loader:
                if self.config['cuda']:
                    input_graph = (Variable(input_1[0].to(self.device, non_blocking=True)),
                                Variable(input_1[1].to(self.device, non_blocking=True)),
                                input_1[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_1[3]])

                    input_transformer = input_2.to(self.device, non_blocking = True)

                else:
                    input_graph = (Variable(input_1[0]),
                                Variable(input_1[1]),
                                input_1[2],
                                input_1[3])
                    input_transformer = input_2
                #print("3rd",os.system('free -h'))
                loss = self._step(model_t, model_g, input_transformer, input_graph)
                loss_total += loss.item() * len(batch_cif_ids)
                total_num += len(batch_cif_ids)
        
            loss_total /= total_num
        #print("4th",os.system('free -h'))
        torch.cuda.empty_cache()
        model_t.train()
        model_g.train()
        return loss_total


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