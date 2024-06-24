import os
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='1113279withHoli_hour_holifix.csv',
                 target='OT', scale=True, 
                 sigma_q=0, sigma_k=1
                 ):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.sigma_q = sigma_q
        self.sigma_k = sigma_k

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_var = pd.read_csv(os.path.join(self.root_path, self.data_path)[:-12] + '.csv_var.csv', header=0)
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, (365+243)*24 - self.seq_len, 365*2*24 - self.seq_len]
        border2s = [(365+243)*24, 365*2*24, 365*3*24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date,format='mixed', dayfirst = True)

        df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)    
        df_stamp['dayofweek'] = df_stamp.date.apply(lambda row:row.dayofweek,1)
        df_stamp['quarter'] = df_stamp.date.apply(lambda row:row.quarter,1)
        df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
        df_stamp['dayofmonth'] = df_stamp.date.apply(lambda row:row.day,1)
        df_stamp['dayofyear'] = df_stamp.date.apply(lambda row:row.timetuple().tm_yday, 1)
        df_stamp['holiday'] = df_raw[['is_public_holiday']][border1:border2]

        data_stamp = df_stamp.drop(['date'], axis = 1).values


        '''jittering'''
        self.data_var = df_var['Variance'].values
        mask = np.isnan(self.data_var)
        self.data_var = self.data_var[~mask]
        
        std_q = np.sqrt(self.data_var) * self.sigma_q
        std_k = np.sqrt(self.data_var) * self.sigma_k

        mu = 0
        eps = np.random.normal(size=std_k.shape)
        z_q = np.add(np.multiply(eps, std_q), mu)
        z_k = np.add(np.multiply(eps, std_k), mu)

        z_q = np.expand_dims(z_q, axis=1)
        z_k = np.expand_dims(z_k, axis=1)

        data_var_q = df_data.values + z_q
        data_var_k = df_data.values + z_k

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_var_q = self.scaler.transform(data_var_q)
            data_var_k = self.scaler.transform(data_var_k)

        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_x_q = data_var_q[border1:border2]
        self.data_x_k = data_var_k[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_var_q = self.data_x_q[s_begin:s_end]
        seq_var_k = self.data_x_k[s_begin:s_end]        
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_var_q, seq_var_k, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

