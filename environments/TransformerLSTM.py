from environments import environment
import torch
import torch.nn as nn
from SpecialLSTM import SpecialLSTM
from consts import *


class TransformerLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config["input_dim"]
        dropout = config["dropout"]
        nhead = config["transformer_nheads"]
        hidden_dim = config["hidden_dim"]
        self.user_vectors = None
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                ).double()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=config["layers"]).double()
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                                 hidden_size=self.hidden_dim,
                                 batch_first=True,
                                 num_layers=self.n_layers,
                                 dropout=dropout)
        seq = [nn.Linear(self.hidden_dim + (input_dim if self.input_twice else 0), self.hidden_dim // 2),
               nn.ReLU(),
               nn.Linear(self.hidden_dim // 2, self.output_dim)]
        if logsoftmax:
            seq += [nn.LogSoftmax(dim=-1)]

        self.output_fc = nn.Sequential(*seq)

        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)

    def forward(self, vectors, **kwargs):
            x = vectors["x"]
            transformer_input = self.fc(x)
            transformer_output = self.transformer(transformer_input)
            lstm_input = self.input_fc(input_vec)
            # lstm_input = lstm_input.reshape(-1, self.hidden_dim)
            # output = self.output_fc(lstm_input)
            lstm_shape = lstm_input.shape
            shape = user_vector.shape
            assert game_vector.shape == shape
            if len(lstm_shape) != len(shape):
                lstm_input = lstm_input.reshape((1,) * (len(shape) - 1) + lstm_input.shape)
            user_vector = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
            game_vector = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))
            lstm_output, (game_vector, user_vector) = self.main_task(lstm_input.contiguous(),
                                                                     (game_vector.contiguous(),
                                                                      user_vector.contiguous()))
            
            lstm_output = self.lstm(transformer_output)
            output = self.output_fc()
            
            return {"output": output}
            

      

