import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder_RNN(nn.Module):
    def __init__(self, e_layers, batch_size,d_model, hidden_size):
        super(Encoder_RNN,self).__init__()
        self.num_layers = e_layers
        self.batch_size = batch_size
        self.input_size = d_model
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, dropout=0.1, batch_first=True).float()
            
    def forward(self,input):
        h0 = Variable(torch.zeros(self.num_layers, input.shape[0], self.hidden_size)).to(input.device)
        output, hidden = self.rnn(input, h0)  

        return output, hidden

class Decoder_RNN(nn.Module):
    def __init__(self, e_layers, batch_size,d_model, hidden_size):
        super(Decoder_RNN,self).__init__()
        self.num_layers = e_layers
        self.batch_size = batch_size
        self.input_size = d_model
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, dropout=0.1, batch_first=True).float()
               
    def forward(self, x, input_hidden):
        input = x.reshape((x.shape[0], 1, self.input_size)) 
        x, hidden_n = self.rnn(input, input_hidden)

        return x, hidden_n

class Encoder_LSTM(nn.Module):
    def __init__(self, e_layers, batch_size, d_model, hidden_size):
        super(Encoder_LSTM,self).__init__()
        self.num_layers = e_layers
        self.batch_size = batch_size
        self.input_size = d_model
        self.hidden_size = hidden_size
        self.cell_size = hidden_size

        self.rnn = nn.LSTM(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, dropout=0.1, batch_first=True, bidirectional=True).float()
            
    def forward(self,input):
        h0 = Variable(torch.zeros(self.num_layers*2, input.shape[0], self.hidden_size)).to(input.device)
        c0 = Variable(torch.zeros(self.num_layers*2, input.shape[0], self.cell_size)).to(input.device)
    
        output, (hidden, cell) = self.rnn(input, (h0,c0)) 

        hidden_1 = torch.cat((hidden[0],hidden[1]),dim=1) 
        hidden_2 = torch.cat((hidden[2],hidden[3]),dim=1) 
        hidden_new = torch.stack((hidden_1,hidden_2),0)

        cell_1 = torch.cat((cell[0],cell[1]),dim=1)
        cell_2 = torch.cat((cell[2],cell[3]),dim=1)
        cell_new = torch.stack((cell_1,cell_2),0)

        return output, hidden_new, cell_new

class Decoder_LSTM(nn.Module):
    def __init__(self, e_layers, batch_size, d_model, hidden_size):
        super(Decoder_LSTM,self).__init__()
        self.num_layers = e_layers
        self.batch_size = batch_size
        self.input_size = d_model
        self.hidden_size = hidden_size*2

        self.rnn = nn.LSTM(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, dropout=0.1, batch_first=True).float()
    
    def forward(self, x, input_hidden, input_cell): 
        input = x.reshape((x.shape[0], 1, self.input_size)) 
        x, (hidden_n, cell_n) = self.rnn(input, (input_hidden, input_cell))

        return x, hidden_n, cell_n

