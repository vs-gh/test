import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.isCuda = isCuda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        
        # initializing weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))
         
    def forward(self, input):
        encoded_input, hidden = self.lstem(input)
        encoded_input = self.relu(encoded_input)
        return encoded_input
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.isCuda = isCuda
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))
       
    def forward(self, encoded_input):
        decoded_output, hidden = self.lstm(encoded_input)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output 

class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTM_AE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, isCuda)
        self.decoder = Decoder(hidden_size, input_size, num_layers, isCuda)
        
    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output