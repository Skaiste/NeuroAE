import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicVAE(nn.Module):
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

    def set_loss_fn_params(self, params):
        self.loss_fn_params = params
    
    def loss(self, x, model_output):
        x_hat, mu, log_var, _ = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        beta = float(self.loss_fn_params.get("beta", 1.0))
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

        return {
            'loss': recon + beta * kld,
            'recon': recon, 
            'kld': kld
        }