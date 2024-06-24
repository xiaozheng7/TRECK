import torch
import torch.nn as nn
import torch.fft as fft
from layers.Embed import DataEmbedding_crafted
from layers.rnn_EncDec import *
import math

from layers.pretrain_encoder import Bottom_encoder
from layers.pretrain_loss import Contrasive_loss

class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1 

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - batch_size, seq_len, d_model
        b, t, _ = input.shape   
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)   
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight) 
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

class HolidayEmbedding(nn.Module):
    def __init__(self, d_model):
        super(HolidayEmbedding, self).__init__()

        holiday_size = 24 
        Embed = nn.Embedding  
        self.holiday_embed = Embed(holiday_size,d_model)
    
    def forward(self, x):
        x = x.to(torch.int64)
        holiday_x = self.holiday_embed(x[:,:,-1])
        return holiday_x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len     
        self.batch_size = configs.batch_size

        self.holiday_embedding = HolidayEmbedding(d_model=configs.d_model)
        self.pattern_holiday_fuse = nn.Linear(configs.d_model*2, configs.d_model)
        self.embed_type = configs.embed_type
       
        '''BiLSTM model'''
        self.encoder = Encoder_LSTM (configs.e_layers, configs.batch_size, configs.d_model, configs.hidden_size)
        self.decoder = Decoder_LSTM (configs.d_layers, configs.batch_size, configs.d_model, configs.hidden_size)

        self.projection = nn.Linear(configs.d_model*2, configs.c_out, bias=True)
        self.dropout = nn.Dropout(configs.dropout)

        if configs.embed_type == 0:
            self.net = Bottom_encoder(              
                input_dims=configs.enc_in,
                output_dims=configs.d_model
            )
            self.contra_loss = Contrasive_loss()
        
        elif configs.embed_type == 1: 
            self.enc_embedding = DataEmbedding_crafted(configs.enc_in, configs.d_model)                                            
            self.dec_embedding = DataEmbedding_crafted(configs.dec_in, configs.d_model)

        self.output_dims = configs.d_model
        self.component_dims = configs.d_model
        self.band_num = configs.band_num
        self.seq_len = configs.seq_len
        self.sfd = nn.ModuleList(
            [BandedFourierLayer(self.output_dims, self.component_dims, b, self.band_num, self.seq_len) for b in range(self.band_num)] 
        )

    def forward(self, x_enc, x_q, x_k, x_mark_enc, x_mark_dec):

        # x_enc.shape = batch_size, seq_len, 1

        if self.embed_type == 0:
            '''holiday embedding'''
            enc_out_holiday = self.holiday_embedding(x_mark_enc)

            x =  self.net(x_enc) # batch_size, seq_len, d_model
            x_q = self.net(x_q)
            x_k = self.net(x_k)

            season_x = []
            for mod in self.sfd:
                out_x = mod(x)  # batch_size, seq_len, d_model
                season_x.append(out_x)
            season_x = season_x[0]
            season_x = self.dropout(season_x)

            season_x_q = []
            for mod in self.sfd:
                out_x = mod(x_q)  
                season_x_q.append(out_x)
            season_x_q = season_x_q[0]
            season_x_q = self.dropout(season_x_q)   

            season_x_k = []
            for mod in self.sfd:
                out_x = mod(x_k)  
                season_x_k.append(out_x)
            season_x_k = season_x_k[0]
            season_x_k = self.dropout(season_x_k) 

            constrasive_loss = self.contra_loss(season_x_q, season_x_k) 
            
            enc_out_pattern = season_x

            enc_out = x + self.pattern_holiday_fuse(torch.cat((enc_out_holiday, enc_out_pattern),dim=-1))
        else:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            constrasive_loss = 0

        enc_out = self.dropout(enc_out) # batch_size, seq_len, d_model
        encoder_output, hidden, cell = self.encoder(enc_out)
        prev_output = enc_out[:,-1,:] # last step as the start token

        for i in range(self.pred_len):

            prev_x_512, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell)
            hidden = prev_hidden
            cell = prev_cell
            prev_x = self.projection(prev_x_512) # batch_size, 1, d_model

            if i == 0:
                targets_dec = prev_x
            else:
                targets_dec = torch.cat((targets_dec, prev_x), dim=1)
            
            if self.embed_type == 0:
                new_input_vol = torch.cat((x_enc[:,targets_dec.shape[1]:,:], targets_dec), dim=1)
                new_input_to_embedding = new_input_vol
                
                with torch.no_grad():
                    enc_out_pattern_1 = self.net(new_input_to_embedding) 
                    season_decoder = []
                    for mod in self.sfd:
                        out_x = mod(enc_out_pattern_1)  
                        season_decoder.append(out_x)
                    season_decoder = season_decoder[0]
                    enc_out_pattern = self.dropout(season_decoder) 

                prev_output =  enc_out_pattern_1[:,-1,:].unsqueeze(1) + self.pattern_holiday_fuse(torch.cat((enc_out_pattern [:,-1,:].unsqueeze(1), self.holiday_embedding(x_mark_dec[:,i,:].unsqueeze(1))),dim=-1))
            else:
                prev_output = self.dec_embedding(prev_x, x_mark_dec[:,i,:].unsqueeze(1))
            
            prev_output = self.dropout(prev_output)

        targets = targets_dec

        return targets, constrasive_loss
