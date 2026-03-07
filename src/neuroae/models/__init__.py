import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.swfcd = None

    def set_loss_fn_params(self, params):
        self.loss_fn_params = params

    def set_swfcd(self, swfcd):
        self.swfcd = swfcd
    