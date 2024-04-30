from environments import environment
import torch
import torch.nn as nn
from SpecialLSTM import SpecialLSTM
from consts import *
from utils.usersvectors import UsersVectors

class multiLayer_TranLSTM(nn.Module):
    def __init__(self, config,logsoftmax = True):
        super().__init__()
        input_dim = config["input_dim"]
        dropout = config["dropout"]
        nhead = config["transformer_nheads"]
        hidden_dim = config["hidden_dim"]
        self.n_layers = config["layers"]
        output_dim = config["output_dim"]
        self.user_vectors = None
        self.input_twice = False
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                ).double()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1).double()
        self.lstm = nn.LSTM(input_size=hidden_dim,
                                 hidden_size=hidden_dim,
                                 batch_first=True,
                                 num_layers=1,
                                 dropout=dropout)
        seq = [nn.Linear(hidden_dim + (input_dim if self.input_twice else 0), hidden_dim // 2),
               nn.ReLU(),
               nn.Linear(hidden_dim // 2, output_dim)]
        if logsoftmax:
            seq += [nn.LogSoftmax(dim=-1)]

        self.output_fc = nn.Sequential(*seq)

        self.user_vectors = UsersVectors(user_dim=hidden_dim, n_layers=1)
        self.game_vectors = UsersVectors(user_dim=hidden_dim, n_layers=1)

    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)

    def forward(self, vectors, **kwargs):
            x = vectors["x"]
            user_vector = vectors["user_vector"]
            game_vector = vectors["game_vector"]
            transformer_input = self.fc(x)
            lstm_output = None
            for i in range(self.n_layers):
                output = []
                for i in range(DATA_ROUNDS_PER_GAME):
                    time_output = self.transformer(transformer_input[:, :i+1].contiguous())[:, -1, :]
                    output.append(time_output)
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
                transformer_input = lstm_output.reshape((1,) * (len(shape) - 1) + lstm_output.shape)
            user_vector = user_vector.reshape(shape)
            game_vector = game_vector.reshape(shape)
            output = self.output_fc(lstm_output)
            if self.training:
                return {"output": output, "game_vector": game_vector, "user_vector": user_vector}
            else:
                return {"output": output, "game_vector": game_vector.detach(), "user_vector": user_vector.detach()}


class multiLayer_TranLSTM_env(environment.Environment):
    def init_model_arc(self, config):
        self.model = multiLayer_TranLSTM(config=config).double()

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
        return {"user_vector": 888, }
             
            

      

