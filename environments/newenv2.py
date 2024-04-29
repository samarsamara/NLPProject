import torch
import torch.nn as nn
from environments import environment
from consts import *
from utils.usersvectors import UsersVectors
import math

class arch5(nn.Module):
    def __init__(self, config, logsoftmax=True):
        super().__init__()
        nhead = config["transformer_nheads"]
        input_dim = config["input_dim"]
        dropout = config["dropout"]
        nhead = config["transformer_nheads"]
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]
        n_layers = config["layers"]
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        ).double()

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        ).double()

        # Layer Normalization after LSTM
        # self.lstm_norm = nn.LayerNorm(hidden_dim).double()

        self.user_vectors = UsersVectors(user_dim=hidden_dim, n_layers=n_layers)
        self.game_vectors = UsersVectors(user_dim=hidden_dim, n_layers=n_layers)

        # Adjusting Transformer's d_model to accommodate concatenated inputs
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # Adjusted to triple the hidden_dim to match concatenated inputs
            nhead=nhead,  # Choose a number of heads that divides evenly into d_model
            dropout=dropout
        ).double()

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=n_layers
        ).double()

        # Output layer remains the same, assuming only hidden_dim is used for prediction
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # Adjusted the input here to match Transformer output
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LogSoftmax(dim=-1) if logsoftmax else nn.Identity()
        ).double()

    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)

    def forward(self, vectors, **kwargs):
        input_vec, game_vector, user_vector = (vectors['x'], vectors['game_vector'], vectors['user_vector'])
        lstm_input = self.input_fc(input_vec)
        lstm_shape = lstm_input.shape
        shape = user_vector.shape
        assert game_vector.shape == shape
        if len(lstm_shape) != len(shape):
            lstm_input = lstm_input.reshape((1,) * (len(shape) - 1) + lstm_input.shape)
        user_vector = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        game_vector = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        lstm_output, (game_vec, user_vec) = self.lstm(lstm_input.contiguous(),
                                                       (game_vector.contiguous(),
                                                        user_vector.contiguous()))

        # Apply normalization to LSTM output
        # lstm_output = self.lstm_norm(lstm_output)

        user_vec = user_vec.reshape(shape)
        game_vec = game_vec.reshape(shape)

       

        output = []
        for i in range(DATA_ROUNDS_PER_GAME):
            time_output = self.transformer_encoder(lstm_output[:, :i + 1].contiguous())[:, -1, :]
            output.append(time_output)
        output = torch.stack(output, 1)
        output = self.output_fc(output)
        if self.training:
            return {"output": output, "game_vector": game_vec, "user_vector": user_vec}
        else:
            return {"output": output, "game_vector": game_vec.detach(), "user_vector": user_vec.detach()}

    def generate_positional_encoding(self, seq_len, hidden_dim):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        positional_encoding = torch.zeros((seq_len, hidden_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding

class arch5_env(environment.Environment):
    def init_model_arc(self, config):
        self.model = arch5(config=config).double()

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            raise NotImplementedError
        output["proba"] = torch.exp(output["output"].flatten())
        return output

    def init_user_vector(self):
        self.currentDM = self.model.init_user()

    def init_game_vector(self):
        self.currentGame = self.model.init_game()

    def get_curr_vectors(self):
        return {"user_vector": 888, }  # You can modify this line accordingly
