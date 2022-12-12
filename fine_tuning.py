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
from torch.nn.modules import transformer
import torch.optim as optim
from torch.utils.data import dataset, DataLoader
from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerRegressor, Transformer, regressoionHead
from model.utils import *
from model.dataset import MOF_ID_Dataset
from datetime import datetime, timedelta
from time import time


def train(dataPath ,modelPath, num_epochs = 30, batch_size = 16, use_ratio=0.05, lr = 0.0005, dropout = 0.2):
    torch.cuda.empty_cache()
    mofdata = np.load(dataPath, allow_pickle=True)
    vocab_path = 'tokenizer/vocab_full.txt'
    tokenizer = MOFTokenizer(vocab_path, model_max_length = 512, padding_side='right')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    if device == torch.device('cpu'):
        num_workers = 0
    else:
        num_workers = 4

    train_data, valid_data = split_data(
        mofdata, train_ratio=0.7, valid_ratio=0.3, use_ratio=use_ratio, randomSeed=None
    )


    train_dataset = MOF_ID_Dataset(train_data, tokenizer)
    valid_dataset = MOF_ID_Dataset(valid_data, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, 
        shuffle=True, pin_memory = False
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, 
        shuffle=True, pin_memory = False
    )

    normalizer = Normalizer(torch.from_numpy(train_dataset.label))
    print("Normalized, mean:", normalizer.mean.numpy(), 'STD:', normalizer.std.numpy())

    # %%
    criterion = nn.MSELoss()
    transformer = Transformer(
        ntoken=4021, d_model=512, nhead=8, 
        d_hid=512, nlayers=4, dropout=0.1
    )
    # Load state dict
    transformer.load_state_dict(torch.load(modelPath))
    model = TransformerRegressor(transformer=transformer, d_model=512).to(device)
    
    # Load model directly
    # transformer = torch.load(modelPath, map_location='cuda:1')
    # model = TransformerRegressor(transformer=transformer.to('cpu'),d_model=512).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    epoch_log = {
        'trainmae':[],
        'trainmse':[],
        'validmae':[],
        'validmse':[]
    }
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        epoch_loss = 0
        trainmse = []
        trainmae = []
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))

            label_norm = normalizer.norm(labels).to(device)

            loss = criterion(outputs, label_norm)
            loss.backward()
            optimizer.step()

            mae_error = mae(normalizer.denorm(outputs.data), labels.to(device))
            mse_error = mse(normalizer.denorm(outputs.data), labels.to(device))
            trainmae.append(mae_error)
            trainmse.append(mse_error)
            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss/len(train_loader)
        trainmae_ave = torch.mean(torch.Tensor(trainmae))
        trainmse_ave = torch.mean(torch.Tensor(trainmse))


        mseList = []
        maeList = []
        with torch.no_grad():
            model.eval()
            for data in valid_loader:
                images, labels = data

                outputs = model(images.to(device))

                label_norm = normalizer.norm(labels).to(device)

                mae_error = mae(normalizer.denorm(outputs.data), labels.to(device)).cpu()
                mse_error = mse(normalizer.denorm(outputs.data), labels.to(device)).cpu()
                maeList.append(mae_error)
                mseList.append(mse_error)
        print('Epoch %d loss:'%(epoch+1), np.round(epoch_loss, 4), 
        '| Train MSE: %.3f' % (trainmse_ave), '| Train MAE: %.3f' % (trainmae_ave),
                '| Valid MSE: %.3f' % (np.mean(mseList)), '| Valid MAE: %.3f' % (np.mean(maeList)))
        epoch_log['trainmae'].append(trainmae_ave)
        epoch_log['trainmse'].append(trainmse_ave)
        epoch_log['validmae'].append(np.mean(maeList))
        epoch_log['validmse'].append(np.mean(mseList))

    epoch_log['trainmae'] = [i.cpu() for i in epoch_log['trainmae']]
    epoch_log['trainmse'] = [i.cpu() for i in epoch_log['trainmse']]
    return epoch_log, model

if __name__ == '__main__':
    n_epoch = 100

    start = time()
    modelPath = os.path.join('training_results/pretraining/30_11_22_54', 'pretrained-model.pt')
    epoch_log, model = train(
        dataPath='data/hMOF_veryClean.npy', modelPath=modelPath, num_epochs=n_epoch, batch_size=128, 
        use_ratio=1, lr = 0.0001, dropout = 0.1
    )
    elapsed = (time() - start)
    print('Training time: %s'%str(timedelta(seconds=elapsed)))
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%H_%M")
    print("Finished training at:", dt_string)
    os.mkdir(os.path.join('training_results/finetuning', dt_string))

    plt.figure(figsize = [4,3], dpi = 150)
    plt.plot(epoch_log['trainmae'], label = 'Training')
    plt.plot(epoch_log['validmae'], label = 'Validation')
    plt.plot(np.arange(0, n_epoch, 1), [0.169]*n_epoch, label = 'CGCNN training')
    plt.plot(np.arange(0, n_epoch, 1), [0.183]*n_epoch, label = 'CGCNN validation')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend(frameon = False)

    plt.savefig(os.path.join('training_results/finetuning', dt_string, 'loss_acc.png'), bbox_inches = 'tight')

    np.savez(
        os.path.join('training_results/finetuning', dt_string, 'training_results.npz'), 
        train_mae = epoch_log['trainmae'],
        train_mse = epoch_log['trainmse'],
        valid_mae = epoch_log['validmae'],
        valid_mse = epoch_log['validmse']
    )
    torch.save(model.state_dict(), os.path.join('training_results/finetuning', dt_string, 'model.pt'))
    shutil.copy2('fine_tuning.py', os.path.join('training_results/finetuning', dt_string))
    print('result_saved')

    plt.close()