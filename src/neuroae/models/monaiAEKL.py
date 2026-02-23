import torch
import torch.nn.functional as F
from monai.networks.nets.autoencoderkl import AutoencoderKL as MonaiAEKL


class AutoencoderKL(MonaiAEKL):
    def forward(self, x):
        # changing this to also get the latent vector
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z)

        # clamp z_sigma to stabilise
        z_sigma = torch.clamp(z_sigma, -10.0, 10.0)
        return reconstruction, z_mu, z_sigma, z
        
    def set_loss_fn_params(self, params):
        if params is not None:
            self.loss_fn_params = {k:v for p in params for k,v in p.items()}
        else:
            self.loss_fn_params = {}
    
    def loss(self, x, model_output):
        x_hat, z_mu, z_sigma, _ = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        beta = float(self.loss_fn_params.get("beta", 1.0))
        # if selected error per feature, we are averaging everything
        if error_per_feature:
            # recon: mean mse loss
            recon = F.mse_loss(x_hat, x, reduction="mean")

        # if selected error per sample, we are summing everything
        else:
            # recon: sum over features per sample, then mean over batch
            recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
            recon = recon.sum(dim=1).mean()

        kld = -0.5 * (1 + z_sigma - z_mu.pow(2) - z_sigma.exp())
        kld = kld.flatten(1).sum(dim=1).mean() / z_sigma.size(1)

        return {
            'loss': recon + beta * kld,
            'recon': recon, 
            'kld': kld
        }
