import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        holiday_size = 24
        hour_size=24
        dayofweek_size=7
        quarter_size=5
        month_size=13
        dayofmonth_size=32
        minute_size=4
        dayofyear_size = 366

        Embed = nn.Embedding 
        self.hour_embed = Embed(hour_size, d_model)
        self.dayofweek_embed = Embed(dayofweek_size,d_model)
        self.quarter_embed = Embed(quarter_size,d_model)
        self.month_embed = Embed(month_size,d_model)
        self.dayofmonth_embed = Embed(dayofmonth_size,d_model)
        self.dayofyear_embed = Embed(dayofyear_size,d_model)
        self.holiday_embed = Embed(holiday_size,d_model)
    
    def forward(self, x):
        x = x.long()
        
        hour_x = self.hour_embed(x[:,:,0])
        dayofweek_x = self.dayofweek_embed(x[:,:,1])
        quarter_x = self.quarter_embed(x[:,:,2])
        month_x = self.month_embed(x[:,:,3])
        dayofmonth_x = self.dayofmonth_embed(x[:,:,4])
        dayofyear_x = self.dayofyear_embed(x[:,:,5])
        holiday_x = self.holiday_embed(x[:,:,6])
        
        return hour_x + dayofweek_x + quarter_x + month_x + dayofmonth_x + dayofyear_x + holiday_x


class DataEmbedding_crafted(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding_crafted, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model) 

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return x

