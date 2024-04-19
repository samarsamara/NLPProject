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
        self.main_task = nn.TransformerEncoder(self.encoder_layer, num_layers=config["layers"]).double()
        

      

