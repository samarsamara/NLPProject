import torch
import torch.nn as nn

class LSTMTransformer(nn.Module):
    def __init__(self,config,logsoftmax = True):
        super().__init__()
        nhead = config["transformer_nheads"]
        input_dim = config["input_dim"]
        dropout = config["dropout"]
        nhead = config["transformer_nheads"]
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]
        n_layers = config["layers"]
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(input_dim * 2, hidden_dim),
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

        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)

        # Adjusting Transformer's d_model to accommodate concatenated inputs
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 3,  # Adjusted to triple the hidden_dim to match concatenated inputs
            nhead=4,  # Choose a number of heads that divides evenly into d_model
            dropout=dropout
        ).double()

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=6
        ).double()

        # Output layer remains the same, assuming only hidden_dim is used for prediction
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # Adjusted the input here to match Transformer output
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LogSoftmax(dim=-1) if logsoftmax else nn.Identity()
        ).double()
        
    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)

    def forward(self, input_vec, game_vector, user_vector):
        lstm_input = self.input_fc(input_vec)

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

        # Concatenation for Transformer input
        combined_input = torch.cat([lstm_out, game_vector, user_vector], dim=-1)
        transformer_out = self.transformer_encoder(combined_input)

        output = self.output_fc(transformer_out)
        return output
