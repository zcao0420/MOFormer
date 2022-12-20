
from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerRegressor, Transformer, regressoionHead
from model.utils import *
from model.dataset import MOF_ID_Dataset
from datetime import datetime, timedelta
from time import time
from torch.utils.data import dataset, DataLoader

import os
import csv
import yaml
import shutil
import sys
import time
import warnings
import numpy as np
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
from model.utils import *
from dataset.dataset_finetune import MOF_ID_Dataset
# from dataset.dataset_finetune import collate_pool, get_train_val_test_loader
#from model.cgcnn_finetune import CrystalGraphConvNet

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_ft.yaml', os.path.join(model_checkpoints_folder, 'config_ft.yaml'))


class FineTune(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time
        log_dir = os.path.join('runs_ft', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        if self.config['task'] == 'classification':
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.MSELoss()


        self.random_seed = self.config['dataloader']['randomSeed']

        self.mofdata = np.load(self.config['dataset']['dataPath'], allow_pickle=True)
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length = 512, padding_side='right')


        #print(self.dataset[0])
        self.train_data, self.valid_data, self.test_data = split_data(
            self.mofdata, valid_ratio = self.config['dataloader']['valid_ratio'], test_ratio = self.config['dataloader']['test_ratio'], 
            use_ratio= self.config['dataloader']['use_ratio'], randomSeed= self.config['dataloader']['randomSeed']
        )
        
        self.train_dataset = MOF_ID_Dataset(data = self.train_data, tokenizer = self.tokenizer, use_ratio= self.config['dataloader']['use_ratio'])
        self.valid_dataset = MOF_ID_Dataset(data = self.valid_data, tokenizer = self.tokenizer, use_ratio= self.config['dataloader']['use_ratio'])
        self.test_dataset = MOF_ID_Dataset(data = self.test_data, tokenizer = self.tokenizer, use_ratio= self.config['dataloader']['use_ratio'])
        
        self.train_loader = DataLoader(
        self.train_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False, 
        shuffle=True, pin_memory = False
        )

        self.valid_loader = DataLoader(
        self.valid_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False, 
        shuffle=False, pin_memory = False
        )

        self.test_loader = DataLoader(
        self.test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False, 
        shuffle=False, pin_memory = False
        )

        self.criterion = nn.MSELoss()

        #obtain target value normalizer
        if self.config['task'] == 'classification':
            self.normalizer = Normalizer(torch.zeros(2))
            self.normalizer.load_state_dict({'mean': 0., 'std': 1.})
        else:
            self.normalizer = Normalizer(torch.from_numpy(self.train_dataset.label))


    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)

        return device

    def train(self):
        self.transformer = Transformer(**self.config['Transformer'])
        # Load state dict
        if self.config['cuda']:
            self.transformer = self.transformer.to(self.device)

        model_transformer = self._load_pre_trained_weights(self.transformer)


        model = TransformerRegressor(transformer=model_transformer, d_model=512).to(self.device)
        
        if self.config['cuda']:
            model = model.to(self.device)

        #model = self._load_pre_trained_weights(model)
        #print(len(model))
        #pytorch_total_params = sum(p.numel() for p in model.parameters if p.requires_grad)
        #print(pytorch_total_params)
        layer_list = []
        for name, param in model.named_parameters():
            if 'fc_out' in name:
                print(name, 'new layer')
                layer_list.append(name)
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        if self.config['optim']['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.config['optim']['lr']*0.2}, {'params': params}],
                 self.config['optim']['init_lr'], momentum=self.config['optim']['momentum'], 
                weight_decay=eval(self.config['optim']['weight_decay'])
            )
        elif self.config['optim']['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                [{'params': base_params, 'lr': self.config['optim']['init_lr']*0.2}, {'params': params}],
                self.config['optim']['init_lr'], weight_decay=eval(self.config['optim']['weight_decay'])
            )
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0

        model.train()

        for epoch_counter in range(self.config['epochs']):
            for bn, (inputs, target) in enumerate(self.train_loader):
                if self.config['cuda']:
                    input_var = inputs.to(self.device)
                else:
                    input_var = inputs.to(self.device)
                
                if self.config['task'] == 'regression':
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target.view(-1).long()
                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output = model(input_var)

                # print(output.shape, target_var.shape)
                loss = self.criterion(output, target_var)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    # self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['task'] == 'classification': 
                    valid_loss, valid_roc_auc = self._validate(model, self.valid_loader)
                    if valid_roc_auc > best_valid_roc_auc:
                        # save the model weights
                        best_valid_roc_auc = valid_roc_auc
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['task'] == 'regression': 
                    valid_loss, valid_mae = self._validate(model, self.valid_loader)
                    if valid_mae < best_valid_mae:
                        # save the model weights
                        best_valid_mae = valid_mae
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                    self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                    valid_n_iter += 1
            
        self.model = model
           
    def _load_pre_trained_weights(self, model):
        # try:
        #     checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
        #     state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
        #     model.load_state_dict(state_dict)
        #     print("Loaded pre-trained model with success.")
        # except FileNotFoundError:
        #     print("Pre-trained weights not found. Training from scratch.")

        try:
            checkpoints_folder = os.path.join(self.config['fine_tune_from'], 'checkpoints')

            load_state = torch.load(os.path.join(checkpoints_folder, 'model_transformer_14.pth'),  map_location=self.config['gpu']) 
 
            # checkpoint = torch.load('model_best.pth.tar', map_location=args.gpu)
            # load_state = checkpoint['state_dict']
            model_state = model.state_dict()

            #pytorch_total_params = sum(p.numel() for p in model_state.parameters if p.requires_grad)
            #print(pytorch_total_params)
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

    def _validate(self, model, valid_loader):
        losses = AverageMeter()
        if self.config['task'] == 'regression':
            mae_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()

        with torch.no_grad():
            model.eval()
            for bn, (inputs, target) in enumerate(valid_loader):
                if self.config['cuda']:
                    input_var = inputs.to(self.device)
                else:
                    input_var = inputs.to(self.device)
                
                if self.config['task'] == 'regression':
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target.view(-1).long()
                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output = model(input_var)
        
                loss = self.criterion(output, target_var)

                if self.config['task'] == 'regression':
                    mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                    losses.update(loss.data.cpu().item(), target.size(0))
                    mae_errors.update(mae_error, target.size(0))
                else:
                    accuracy, precision, recall, fscore, auc_score = \
                        class_eval(output.data.cpu(), target)
                    losses.update(loss.data.cpu().item(), target.size(0))
                    accuracies.update(accuracy, target.size(0))
                    precisions.update(precision, target.size(0))
                    recalls.update(recall, target.size(0))
                    fscores.update(fscore, target.size(0))
                    auc_scores.update(auc_score, target.size(0))
            
            if self.config['task'] == 'regression':
                print('Test: [{0}/{1}], '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    bn, len(self.valid_loader), loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}], '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Accu {accu.val:.3f} ({accu.avg:.3f}), '
                      'Precision {prec.val:.3f} ({prec.avg:.3f}), '
                      'Recall {recall.val:.3f} ({recall.avg:.3f}), '
                      'F1 {f1.val:.3f} ({f1.avg:.3f}), '
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    bn, len(self.valid_loader), loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))
        
        model.train()

        if self.config['task'] == 'regression':
            print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
            return losses.avg, mae_errors.avg
        else:
            print('AUC {auc.avg:.3f}'.format(auc=auc_scores))
            return losses.avg, auc_scores.avg

    
    def test(self):
        # test steps
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        print(model_path)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        losses = AverageMeter()
        if self.config['task'] == 'regression':
            mae_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()
        
        test_targets = []
        test_preds = []
        test_cif_ids = []

        with torch.no_grad():
            self.model.eval()
            for bn, (inputs, target) in enumerate(self.test_loader):
                if self.config['cuda']:
                    input_var = inputs.to(self.device)
                else:
                    input_var = inputs.to(self.device)
                
                if self.config['task'] == 'regression':
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target.view(-1).long()
                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output = self.model(input_var)
        
                loss = self.criterion(output, target_var)

                if self.config['task'] == 'regression':
                    mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                    losses.update(loss.data.cpu().item(), target.size(0))
                    mae_errors.update(mae_error, target.size(0))
                    
                    test_pred = self.normalizer.denorm(output.data.cpu())
                    test_target = target
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += test_target.view(-1).tolist()
                else:
                    accuracy, precision, recall, fscore, auc_score = \
                        class_eval(output.data.cpu(), target)
                    losses.update(loss.data.cpu().item(), target.size(0))
                    accuracies.update(accuracy, target.size(0))
                    precisions.update(precision, target.size(0))
                    recalls.update(recall, target.size(0))
                    fscores.update(fscore, target.size(0))
                    auc_scores.update(auc_score, target.size(0))
                   
                    test_pred = torch.exp(output.data.cpu())
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()

            
            if self.config['task'] == 'regression':
                print('Test: [{0}/{1}], '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    bn, len(self.valid_loader), loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}], '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Accu {accu.val:.3f} ({accu.avg:.3f}), '
                      'Precision {prec.val:.3f} ({prec.avg:.3f}), '
                      'Recall {recall.val:.3f} ({recall.avg:.3f}), '
                      'F1 {f1.val:.3f} ({f1.avg:.3f}), '
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    bn, len(self.valid_loader), loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

        with open(os.path.join(self.writer.log_dir, 'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
        
        self.model.train()

        if self.config['task'] == 'regression':
            print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
            return losses.avg, mae_errors.avg
        else:
            print('AUC {auc.avg:.3f}'.format(auc=auc_scores))
            return losses.avg, auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    Parameters
    ----------
    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    

if __name__ == "__main__":
    config = yaml.load(open("config_ft.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    if 'MOF-TOY' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'MOF-TOY'
    if 'hMOF' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'hMOF'
    elif 'fermi' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'fermi'
    elif 'Is_Metal_cifs' in config['dataset']['data_name']:
        config['task'] = 'classification'
        task_name = 'Is_Metal_cifs'
    elif 'MP-formation-energy' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'MP_energy'
    elif 'lanths' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'lanths'
    elif 'jdft2d' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'jdft2d'
    elif 'phonons' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'phonons'
    elif 'perovskites' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'perovskites'
    elif 'FE' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'FE'
    elif 'GVRH' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'GVRH'
    elif 'HOIP' in config['dataset']['data_name']:
        config['task'] = 'regression'
        task_name = 'HOIP'

    fine_tune = FineTune(config)
    fine_tune.train()
    loss, metric = fine_tune.test()

    import pandas as pd
    ftf = config['fine_tune_from'].split('/')[-1]
    seed = config['dataloader']['randomSeed']
    fn = '{}_{}_200epochs_{}.csv'.format(ftf, task_name,seed)
    print(fn)
    df = pd.DataFrame([[loss, metric.item()]])
    df.to_csv(
        os.path.join('experiments', fn),
        mode='a', index=False, header=False
    )