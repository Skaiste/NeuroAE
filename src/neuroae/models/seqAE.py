import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialAE(nn.Module):
    """
    Input per sample: (T, R)  where
      T = timepoints, R = regions
    Internally uses batch-first: (B, T, R)
    """
    def __init__(self, regions: int, hidden_dim: int = 256, latent_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.0, cell: str = "lstm"):
        super().__init__()
        self.regions = regions
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.cell = cell.lower()

        rnn_cls = nn.LSTM if self.cell == "lstm" else nn.GRU

        # Encoder RNN reads the sequence and outputs last hidden state
        self.encoder_rnn = rnn_cls(
            input_size=regions,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Map encoder hidden -> latent z
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # Map latent -> decoder initial hidden
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        # Decoder RNN produces a sequence of hidden states
        self.decoder_rnn = rnn_cls(
            input_size=regions,  # we’ll feed zeros or a learned “start token” in region-space
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Map decoder hidden states -> reconstructed regions
        self.output_layer = nn.Linear(hidden_dim, regions)

        # Optional: learned start token in region-space
        self.start_token = nn.Parameter(torch.zeros(1, 1, regions))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, R)
        if self.cell == "lstm":
            _, (h_n, _) = self.encoder_rnn(x)   # h_n: (L, B, H)
        else:
            _, h_n = self.encoder_rnn(x)        # h_n: (L, B, H)

        h_last = h_n[-1]                        # (B, H)
        z = self.to_latent(h_last)              # (B, Z)
        return z

    def decode(self, z: torch.Tensor, T: int) -> torch.Tensor:
        # z: (B, Z)
        B = z.shape[0]
        h0_last = torch.tanh(self.from_latent(z))  # (B, H)

        # Build initial hidden for all layers
        h0 = h0_last.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (L, B, H)
        if self.cell == "lstm":
            c0 = torch.zeros_like(h0)

        # Input to decoder: start token then zeros (you can also teacher-force)
        dec_in = torch.zeros(B, T, self.regions, device=z.device, dtype=z.dtype)
        dec_in[:, :1, :] = self.start_token  # first step has learned token

        if self.cell == "lstm":
            dec_h, _ = self.decoder_rnn(dec_in, (h0, c0))  # (B, T, H)
        else:
            dec_h, _ = self.decoder_rnn(dec_in, h0)

        x_hat = self.output_layer(dec_h)  # (B, T, R)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, R)
        z = self.encode(x)
        x_hat = self.decode(z, T=x.shape[1])
        return x_hat, z
    
    # a placeholder since the loss function doesn't have any parameters
    def set_loss_fn_params(self, params):
        pass

    def loss(self, x, model_output):
        return {'loss': F.mse_loss(model_output[0], x)}
