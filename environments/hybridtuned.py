import torch
import torch.nn as nn
from environments import environment
from consts import *
from utils.usersvectors import UsersVectors
class myhybrid(nn.Module):
    def __init__(self,config,logsoftmax = True):
        super().__init__()
        nhead = config["transformer_nheads"]
        input_dim = config["input_dim"]
        dropout = config["dropout"]
        nhead = config["transformer_nheads"]
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]
        n_layers = config["layers"]

        self.input_fc = nn.Sequential(nn.Linear(input_dim, input_dim * 2),
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(input_dim * 2,hidden_dim),
                                      nn.Dropout(dropout),
                                      nn.ReLU())

        self.main_task = nn.LSTM(input_size=hidden_dim,
                                 hidden_size=hidden_dim,
                                 batch_first=True,
                                 num_layers=n_layers,
                                 dropout=dropout)

        seq = [nn.Linear(hidden_dim,hidden_dim // 2),
               nn.ReLU()]

        self.output_fc = nn.Sequential(*seq)
        ##building transformer
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                ).double()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.main_task1 = nn.TransformerEncoder(self.encoder_layer, num_layers=config["layers"]).double()

        self.main_task_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                                  nn.ReLU()).double()
        self.pred = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                                  nn.ReLU(),
                                                  nn.Linear(hidden_dim//2, 2),
                                                  nn.LogSoftmax(dim=-1)).double()
        
        self.user_vectors = UsersVectors(user_dim=hidden_dim, n_layers=n_layers)
        self.game_vectors = UsersVectors(user_dim=hidden_dim, n_layers=n_layers)

    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)

    def forward(self, vectors,**kwargs):
        input_vec, game_vector, user_vector = (vectors['x'],vectors['game_vector'],vectors['user_vector'])
        x = self.fc(input_vec)
        output = []
        for i in range(DATA_ROUNDS_PER_GAME):
            time_output = self.main_task1(x[:, :i+1].contiguous())[:, -1, :]
            output.append(time_output)
        output = torch.stack(output, 1)
        output = self.main_task_classifier(output)
        lstm_input = self.input_fc(input_vec)
        lstm_shape = lstm_input.shape
        shape = user_vector.shape
        assert game_vector.shape == shape
        if len(lstm_shape) != len(shape):
            lstm_input = lstm_input.reshape((1,) * (len(shape) - 1) + lstm_input.shape)
        user_vector = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        game_vector = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        lstm_output, (game_vec, user_vec) = self.main_task(lstm_input.contiguous(),
                                                                 (game_vector.contiguous(),
                                                                  user_vector.contiguous()))
        concatenated_output = torch.cat((lstm_output, output), dim=2)
        hidden_dim = concatenated_output.size(2)
        pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.LogSoftmax(dim=-1)
        ).double()

        # Apply the fully connected layers to the concatenated output
        final_output = pred(concatenated_output.view(-1, hidden_dim))
        user_vec = user_vec.reshape(shape)
        game_vec = game_vec.reshape(shape)

        if self.training:
            return {"output": final_output, "game_vector": game_vec, "user_vector": user_vec}
        else:
            return {"output": final_output, "game_vector": game_vec.detach(), "user_vector": user_vec.detach()}
class myhyprid_env(environment.Environment):
        def init_model_arc(self, config):
            self.model = myhybrid(config=config).double()
    
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
            return {"user_vector": 888, }

