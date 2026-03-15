# implement a 1D AE to encode a 2D AE latent space
import torch
from collections.abc import Sequence
import torch.nn.functional as F

from . import ModelBase
from .linear import LinearAE
from .autoencoderkl import AutoencoderKLv3

class AEKLv3pp(ModelBase):
    def __init__(self, 
                 timepoints: int,
                 latent2_dim: int = 64,
                 in_channels: int = 1,
                 num_res_blocks = (2, 2, 2, 2),
                 channels = (32, 64, 64, 64),
                 attention_levels = (False, False, True, True),
                 latent_channels = 3,
                 norm_num_groups = 32,
                 norm_eps = 1e-6,
                 with_encoder_nonlocal_attn = True,
                 with_decoder_nonlocal_attn = True):
        super().__init__()

        self.convAE = AutoencoderKLv3(
            spatial_dims=1,
            in_channels=in_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            channels=channels,
            attention_levels=attention_levels,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
            decoder_latent_channels=latent_channels*2)
        
        self.linearAE = LinearAE(latent_channels * timepoints, latent2_dim)

    def forward(self, x):
        # changing this to also get the latent vector
        z_mu, z_sigma = self.convAE.encode(x)
        z1 = self.convAE.sampling(z_mu, z_sigma)
        z1_flat = z1.reshape(z1.shape[0], -1)

        recon2, z2 = self.linearAE(z1_flat)

        recon2_2d = recon2.reshape(z1.shape)
        z1_with_recon = torch.cat((z1, recon2_2d), 1)
        
        recon1 = self.convAE.decode(z1_with_recon)

        return recon1, z_mu, z_sigma, z1
    
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
        
        

