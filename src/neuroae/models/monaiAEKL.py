import torch.nn.functional as F
from monai.networks.nets.autoencoderkl import AutoencoderKL as MonaiAEKL

class AutoencoderKL(MonaiAEKL):
    def set_loss_fn_params(self, params):
        if params is not None:
            self.loss_fn_params = {k:v for p in params for k,v in p.items()}
        else:
            self.loss_fn_params = {}
    
    def loss(self, x, model_output):
        x_hat, mu, log_var = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        kld_weight = float(self.loss_fn_params.get("kld_weight", 1.0))
        # if selected error per feature, we are averaging everything
        if error_per_feature:
            # recon: mean mse loss
            recon = F.mse_loss(x_hat, x, reduction="mean")

            # KL: mean over batch, then mean over latent dims
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.flatten(1).sum(dim=1).mean() / log_var.size(1)

        # if selected error per sample, we are summing everything
        else:
            # recon: sum over features per sample, then mean over batch
            recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
            recon = recon.sum(dim=1).mean()

            # kld: sum over latent dims per sample, then mean over batch
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.flatten(1).sum(dim=1).mean() / log_var.size(1)

        return {
            'loss': recon + kld_weight * kld,
            'recon': recon, 
            'kld': kld
        }