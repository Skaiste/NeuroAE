import torch.nn as nn
import torch.nn.functional as F


class LinearAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    # a placeholder since the loss function doesn't have any parameters
    def set_loss_fn_params(self, params):
        pass

    def loss(self, x, model_output):
        return {'loss': F.mse_loss(model_output[0], x)}
    
