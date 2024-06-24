from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import rnn, bilstm
from utils.tools import EarlyStopping, adjust_learning_rate, log_string
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'rnn':rnn,
            'rnn_bilstm':bilstm,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_var_q, batch_var_k, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_var_q = batch_var_q.float().to(self.device)
                batch_var_k = batch_var_k.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
    
                outputs, contrasive_loss = self.model(batch_x, batch_var_q, batch_var_k, batch_x_mark, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss_1 = criterion(pred, true)
                loss_2 = contrasive_loss

                loss = loss_1

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def train(self, setting, log):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_var_q, batch_var_k, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_var_q = batch_var_q.float().to(self.device)
                batch_var_k = batch_var_k.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, contrasive_loss = self.model(batch_x, batch_var_q, batch_var_k, batch_x_mark, batch_y_mark)
     
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss_1 = criterion(outputs, batch_y) 
                loss_2 = contrasive_loss

                if self.args.embed_type == 0:
                    loss = loss_1 + self.args.beta * loss_2
                else:
                    loss = loss_1

                train_loss.append(loss_1.item())

                if (i + 1) % 100 == 0:  
                    if self.args.embed_type == 0:
                        log_string(log,"\titers: {0}, epoch: {1} | loss_MSE: {2:.4f} | loss_contra: {3:.4f} | total_loss: {4:.4f}".format(i + 1, epoch + 1, loss_1.item(), loss_2.item(), loss.item()))
                    else:
                        log_string(log,"\titers: {0}, epoch: {1} | loss_MSE: {2:.4f} | loss_contra: NA | total_loss: {3:.4f}".format(i + 1, epoch + 1, loss_1.item(), loss.item()))                        
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    log_string(log,'\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
       
                loss.backward()
                model_optim.step()

            log_string(log,"Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            log_string(log,"Epoch: {0}, Steps: {1} | Train Loss (only MSE): {2:.7f} Vali Loss (only MSE): {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                log_string(log,"Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, log, exp_path, test=1):
        test_data, test_loader = self._get_data(flag='test')
        time_now = time.time()

        if test:
            log_string(log,'loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = exp_path +  '/'

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_var_q, batch_var_k, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # print (batch_x.shape)
                
                batch_x = batch_x.float().to(self.device)
                batch_var_q = batch_var_q.float().to(self.device)
                batch_var_k = batch_var_k.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, contrasive_loss = self.model(batch_x, batch_var_q, batch_var_k, batch_x_mark, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        log_string(log,"all_testing time:{}".format(time.time() - time_now))            
        preds = np.concatenate(preds, axis=0).squeeze()
        trues = np.concatenate(trues, axis=0).squeeze()

        preds = preds.reshape(-1,self.args.pred_len)
        trues = trues.reshape(-1,self.args.pred_len)

        print (preds.shape)
        print (trues.shape)

        preds = test_data.inverse_transform(preds)
        trues = test_data.inverse_transform(trues)

        mae, mse, rmse, mape100, mape, mspe, rse, corr= metric(preds, trues)

        log_string(log,f'test shape: (in [num_samples, pred_len, c_out]): {preds.shape}, {trues.shape}') 
        log_string(log, '                MAE\t\tRMSE\t\tMAPE\t\tMAPE_100')
        log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%\t\t%.2f%%' %
               (mae, rmse, mape*100, mape100*100))

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

