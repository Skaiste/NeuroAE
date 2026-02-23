import torch
import torch.nn as nn
from typing import Sequence, Optional



class DeterministicAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (4096, 1024, 256),
        latent_dim: int = 64,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        final_activation: Optional[str] = None,  # None | "sigmoid" | "tanh"
    ):
        super().__init__()
        assert input_dim > 0
        assert latent_dim > 0
        assert len(hidden_dims) >= 1

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = latent_dim
        self.final_activation = final_activation

        # for loss
        self.criterion = nn.MSELoss()

        def block(in_f: int, out_f: int) -> nn.Sequential:
            layers = [nn.Linear(in_f, out_f)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_f))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            return nn.Sequential(*layers)

        # ---------- Encoder ----------
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(block(prev, h))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.to_latent = nn.Linear(prev, latent_dim)

        # ---------- Decoder ----------
        self.from_latent = nn.Linear(latent_dim, hidden_dims[-1])
        dec_layers = []
        rev_hidden = list(hidden_dims[::-1])  # e.g., 256, 1024, 4096
        prev = rev_hidden[0]
        for h in rev_hidden[1:]:
            dec_layers.append(block(prev, h))
            prev = h
        self.decoder = nn.Sequential(*dec_layers)
        self.to_recon = nn.Linear(prev, input_dim)

        if final_activation not in (None, "sigmoid", "tanh"):
            raise ValueError("final_activation must be None, 'sigmoid', or 'tanh'.")

        self._final_act = {
            None: nn.Identity(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }[final_activation]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x shape (B, {self.input_dim}), got {tuple(x.shape)}")
        h = self.encoder(x)
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected z shape (B, {self.latent_dim}), got {tuple(z.shape)}")
        h = torch.relu(self.from_latent(z))
        h = self.decoder(h)
        x_hat = self.to_recon(h)
        x_hat = self._final_act(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    # a placeholder since the loss function doesn't have any parameters
    def set_loss_fn_params(self, params):
        pass

    def loss(self, x, model_output):
        return {'loss': self.criterion(model_output[0], x)}