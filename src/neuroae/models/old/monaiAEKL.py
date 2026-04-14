import torch
import torch.nn.functional as F
from monai.networks.nets.autoencoderkl import AutoencoderKL as MonaiAEKL


class AutoencoderKLv1(MonaiAEKL):
    def __init__(self, spatial_dims, in_channels = 1, out_channels = 1, num_res_blocks = ..., channels = ..., attention_levels = ..., latent_channels = 3, norm_num_groups = 32, norm_eps = 0.000001, with_encoder_nonlocal_attn = True, with_decoder_nonlocal_attn = True, use_checkpoint = False, use_convtranspose = False, include_fc = True, use_combined_linear = False, use_flash_attention = False):
        super().__init__(spatial_dims, in_channels, out_channels, num_res_blocks, channels, attention_levels, latent_channels, norm_num_groups, norm_eps, with_encoder_nonlocal_attn, with_decoder_nonlocal_attn, use_checkpoint, use_convtranspose, include_fc, use_combined_linear, use_flash_attention)
        self.swfcd = None

    def forward(self, x):
        # changing this to also get the latent vector
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z)

        # clamp z_sigma to stabilise
        z_sigma = torch.clamp(z_sigma, -10.0, 10.0)
        return reconstruction, z_mu, z_sigma, z
        
    def set_loss_fn_params(self, params):
        self.loss_fn_params = params

    def set_swfcd(self, swfcd):
        self.swfcd = swfcd
    
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
    