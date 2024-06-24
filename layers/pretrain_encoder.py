import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x


class Bottom_encoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.feature_extractor = TokenEmbedding(c_in=input_dims, d_model=output_dims)

    def forward(self, x): 
        x = self.feature_extractor(x)  
        return x
