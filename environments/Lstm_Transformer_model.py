import torch
import torch.nn as nn

class LSTMTransformer(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dropout, logsoftmax=True):
        super().__init__()
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

        # Adjusting Transformer's d_model to accommodate concatenated inputs
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 3,  # Adjusted to triple the hidden_dim to match concatenated inputs
            nhead=8,  # Choose a number of heads that divides evenly into d_model
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

    def forward(self, input_vec, game_vector, user_vector):
        x = self.input_fc(input_vec)

        # LSTM processing
        lstm_out, (hn, cn) = self.lstm(x)

        # Concatenation for Transformer input
        combined_input = torch.cat([lstm_out, game_vector, user_vector], dim=-1)
        transformer_out = self.transformer_encoder(combined_input)

        output = self.output_fc(transformer_out)
        return output
