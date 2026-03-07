import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelBase


class BasicVAE(ModelBase):
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dims, latent_dim):
            super().__init__()
            if type(hidden_dims) == int:
                hidden_dims = [hidden_dims]

            layers = []
            last_dim = input_dim
            for dim in hidden_dims:
                layers.append(nn.Linear(last_dim, dim))
                layers.append(nn.LeakyReLU(0.2))
                last_dim = dim
            self.fc = nn.Sequential(*layers)
            self.fc_mean = nn.Linear(last_dim, latent_dim)
            self.fc_logvar = nn.Linear(last_dim, latent_dim)

        def forward(self, x):
            h = self.fc(x)
            mean = self.fc_mean(h)
            log_var = self.fc_logvar(h)
            # clamp log_var to stabilise
            log_var = torch.clamp(log_var, -10.0, 10.0)
            return mean, log_var
    
    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dims, output_dim):
            super().__init__()

            if type(hidden_dims) == int:
                hidden_dims = [hidden_dims]

            layers = []
            last_dim = latent_dim
            # assumint that the hidden dimensions are provided in the encoder order
            for dim in hidden_dims[::-1]:
                layers.append(nn.Linear(last_dim, dim))
                layers.append(nn.LeakyReLU(0.2))
                last_dim = dim
            layers.append(nn.Linear(last_dim, output_dim))
            self.fc = nn.Sequential(*layers)

        def forward(self, x):
            return self.fc(x)

    def __init__(self, 
        input_dim=784, 
        hidden_dims=[1024, 512, 256, 128], 
        latent_dim=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(BasicVAE, self).__init__()
        self.encoder = self.Encoder(input_dim, hidden_dims, latent_dim).to(device)
        self.decoder = self.Decoder(latent_dim, hidden_dims, input_dim).to(device)
        self.device = device
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.swfcd = None

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def reset_decoder(self):
        self.decoder = self.Decoder(self.latent_dim, self.hidden_dims, self.input_dim).to(self.device)

    def reparameterize(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)         # sampling epsilon        
        z = mean + var*epsilon                                  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        # clamp log_var to prevent numerical instability
        log_var = torch.clamp(log_var, -10.0, 10.0)
        z = self.reparameterize(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var, z
    
    def loss(self, x, model_output):
        x_hat, mu, log_var, _ = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        beta = self.loss_fn_params.get("beta", 1.0)
        # if selected error per feature, we are averaging everything
        if error_per_feature:
            # recon: mean mse loss
            recon = F.mse_loss(x_hat, x, reduction="mean")

            # KL: mean over batch, then mean over latent dims
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)

        # if selected error per sample, we are summing everything
        else:
            # recon: sum over features per sample, then mean over batch
            recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
            recon = recon.sum(dim=1).mean()

            # kld: sum over latent dims per sample, then mean over batch
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)

        loss = {
            'loss': recon + beta * kld,
            'recon': recon, 
            'kld': kld
        }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, x_hat)
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss['swfcd_rmse'] = swfcd['rmse']
            loss['loss'] += swfcd_beta * swfcd['rmse']

        return loss


class BasicVAETimeShared(ModelBase):
    class Encoder(nn.Module):
        def __init__(self, feature_dim, hidden_dims, latent_dim):
            super().__init__()
            if type(hidden_dims) == int:
                hidden_dims = [hidden_dims]

            layers = []
            last_dim = feature_dim
            for dim in hidden_dims:
                layers.append(nn.Linear(last_dim, dim))
                layers.append(nn.LeakyReLU(0.2))
                last_dim = dim
            self.fc = nn.Sequential(*layers)
            self.fc_mean = nn.Linear(last_dim, latent_dim)
            self.fc_logvar = nn.Linear(last_dim, latent_dim)

        def forward(self, x):
            # x shape: (B, T, F)
            h = self.fc(x)
            mean = self.fc_mean(h)
            log_var = self.fc_logvar(h)
            log_var = torch.clamp(log_var, -10.0, 10.0)
            return mean, log_var

    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dims, feature_dim):
            super().__init__()

            if type(hidden_dims) == int:
                hidden_dims = [hidden_dims]

            layers = []
            last_dim = latent_dim
            for dim in hidden_dims[::-1]:
                layers.append(nn.Linear(last_dim, dim))
                layers.append(nn.LeakyReLU(0.2))
                last_dim = dim
            layers.append(nn.Linear(last_dim, feature_dim))
            self.fc = nn.Sequential(*layers)

        def forward(self, z):
            # z shape: (B, T, L)
            return self.fc(z)

    def __init__(self,
        input_dim=784,
        timepoint_dim=1,
        hidden_dims=[256, 128],
        latent_dim=8,
        input_layout="feature_time",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(BasicVAETimeShared, self).__init__()
        self.input_dim = int(input_dim)
        self.timepoint_dim = int(timepoint_dim)
        if self.timepoint_dim <= 0:
            raise ValueError("timepoint_dim must be > 0.")
        if self.input_dim % self.timepoint_dim != 0:
            raise ValueError(
                f"input_dim ({self.input_dim}) must be divisible by "
                f"timepoint_dim ({self.timepoint_dim})."
            )

        self.feature_dim = self.input_dim // self.timepoint_dim
        self.hidden_dims = hidden_dims
        self.latent_per_timepoint = int(latent_dim)
        self.latent_dim = self.latent_per_timepoint * self.timepoint_dim
        self.input_layout = str(input_layout)
        if self.input_layout not in {"feature_time", "time_feature"}:
            raise ValueError("input_layout must be one of {'feature_time', 'time_feature'}.")
        self.device = device

        self.encoder = self.Encoder(
            feature_dim=self.feature_dim,
            hidden_dims=hidden_dims,
            latent_dim=self.latent_per_timepoint,
        ).to(device)
        self.decoder = self.Decoder(
            latent_dim=self.latent_per_timepoint,
            hidden_dims=hidden_dims,
            feature_dim=self.feature_dim,
        ).to(device)

    def _reshape_input(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x shape (B, {self.input_dim}), got {tuple(x.shape)}")
        if self.input_layout == "feature_time":
            # Flattened input is interpreted as (B, F, T); transpose to (B, T, F).
            x_time = x.reshape(x.shape[0], self.feature_dim, self.timepoint_dim).transpose(1, 2)
        else:
            # Flattened input is interpreted as (B, T, F).
            x_time = x.reshape(x.shape[0], self.timepoint_dim, self.feature_dim)
        return x_time

    def _flatten_recon(self, x_time):
        if self.input_layout == "feature_time":
            # (B, T, F) -> (B, F, T) -> (B, D)
            return x_time.transpose(1, 2).reshape(x_time.shape[0], self.input_dim)
        # (B, T, F) -> (B, D)
        return x_time.reshape(x_time.shape[0], self.input_dim)

    def _flatten_latent(self, z_time):
        # (B, T, L) -> (B, T*L)
        return z_time.reshape(z_time.shape[0], self.latent_dim)

    def _reshape_latent(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected z shape (B, {self.latent_dim}), got {tuple(z.shape)}")
        return z.reshape(z.shape[0], self.timepoint_dim, self.latent_per_timepoint)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def reset_decoder(self):
        self.decoder = self.Decoder(
            latent_dim=self.latent_per_timepoint,
            hidden_dims=self.hidden_dims,
            feature_dim=self.feature_dim,
        ).to(self.device)

    def reparameterize(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        x_time = self._reshape_input(x)
        mean_time, log_var_time = self.encoder(x_time)
        log_var_time = torch.clamp(log_var_time, -10.0, 10.0)

        mean = self._flatten_latent(mean_time)
        log_var = self._flatten_latent(log_var_time)
        z = self.reparameterize(mean, torch.exp(0.5 * log_var))

        z_time = self._reshape_latent(z)
        x_hat_time = self.decoder(z_time)
        x_hat = self._flatten_recon(x_hat_time)
        return x_hat, mean, log_var, z

    def loss(self, x, model_output):
        x_hat, mu, log_var, _ = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        beta = float(self.loss_fn_params.get("beta", 1.0))
        if error_per_feature:
            recon = F.mse_loss(x_hat, x, reduction="mean")
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)
        else:
            recon = F.mse_loss(x_hat, x, reduction="none")
            recon = recon.sum(dim=1).mean()
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)

        loss = {
            'loss': recon + beta * kld,
            'recon': recon, 
            'kld': kld
        }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, x_hat)
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss['swfcd_rmse'] = swfcd['rmse']
            loss['loss'] += swfcd_beta * swfcd['rmse']

        return loss
