import torch
import torch.nn as nn
from environments import environment
from consts import *
from utils.usersvectors import UsersVectors

class TransformerLSTMWithSelfAttention(nn.Module):
    def __init__(self, config, logsoftmax=True):
        super().__init__()
        input_dim = config["input_dim"]
        dropout = config["dropout"]
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]
        n_layers = config["layers"]
        self.user_vectors = None

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        ).double()

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=n_layers,
            dropout=dropout
        ).double()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,  # Adjust as needed
            dropout=dropout
        ).double()

        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LogSoftmax(dim=-1) if logsoftmax else nn.Identity()
        ).double()

        self.user_vectors = UsersVectors(user_dim=hidden_dim, n_layers=n_layers)
        self.game_vectors = UsersVectors(user_dim=hidden_dim, n_layers=n_layers)

    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)

    def forward(self, vectors, **kwargs):
        x = vectors["x"]
        user_vector = vectors["user_vector"]
        game_vector = vectors["game_vector"]

        transformer_input = self.fc(x)

        output = []
        for i in range(DATA_ROUNDS_PER_GAME):
            current_time_input = transformer_input[:, :i+1].contiguous()
            current_time_output, _ = self.self_attention(current_time_input, current_time_input, current_time_input)
            current_time_output = current_time_output[:, -1, :]
            output.append(current_time_output)

        lstm_input = torch.stack(output, 1)
        lstm_shape = lstm_input.shape
        shape = user_vector.shape
        assert game_vector.shape == shape

        if len(lstm_shape) != len(shape):
            lstm_input = lstm_input.reshape((1,) * (len(shape) - 1) + lstm_input.shape)

        user_vector = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        game_vector = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))

        lstm_output, (game_vector, user_vector) = self.lstm(lstm_input.contiguous(),
                                                             (game_vector.contiguous(),
                                                              user_vector.contiguous()))

        user_vector = user_vector.reshape(shape)
        game_vector = game_vector.reshape(shape)

        output = self.output_fc(lstm_output)

        if self.training:
            return {"output": output, "game_vector": game_vector, "user_vector": user_vector}
        else:
            return {"output": output, "game_vector": game_vector.detach(), "user_vector": user_vector.detach()}


class TransformerLSTMWithSelfAttentionEnv(environment.Environment):
    def init_model_arc(self, config):
        self.model = TransformerLSTMWithSelfAttention(config=config).double()

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            output = self.model({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        if update_vectors:
            self.currentDM = output["user_vector"]
            self.currentGame = output["game_vector"]
        return output

    def init_user_vector(self):
        self.currentDM = self.model.init_user()

    def init_game_vector(self):
        self.currentGame = self.model.init_game()

    def get_curr_vectors(self):
        return {"user_vector": 888}
