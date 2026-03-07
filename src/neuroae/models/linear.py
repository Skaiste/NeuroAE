import torch.nn as nn
import torch.nn.functional as F

from . import ModelBase


class LinearAE(ModelBase):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def loss(self, x, model_output):
        loss = {
            'loss': F.mse_loss(model_output[0], x)
        }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, model_output[0])
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss['swfcd_rmse'] = swfcd['rmse']
            loss['loss'] += swfcd_beta * swfcd['rmse']

        return loss

