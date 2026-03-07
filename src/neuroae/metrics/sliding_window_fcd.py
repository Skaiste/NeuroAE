import numpy as np
from neuronumba.observables.sw_fcd import SwFCD as Numba_SwFCD

from ..utils.np_utils import to_numpy
from ..load_data import ADNIDataset


class SwFCD:
    def __init__(self, dataset:ADNIDataset, window_size:int, window_step:int):
        self.dataset = dataset
        self.swfcd = Numba_SwFCD(window_size=window_size, window_step=window_step)

    def ensure_correct_dim(self, x):    # --> of shape (B, T, R)
        if self.dataset.fc_input: # this type of data cannot be used for swfcd
            return None
        
        x_np = to_numpy(x)

        if self.dataset.timepoints_as_samples: # current shape (B x T, R)
            x_np = x_np.reshape(-1, self.dataset.original_shape[0], self.dataset.original_shape[1]) # to (B, T, R)
            return x_np        
        
        if self.dataset.flatten: # current shape (B, R x T) or (B, T x R)
            x_np = x_np.reshape(-1, self.dataset.original_shape[0], self.dataset.original_shape[1])
        if not self.dataset.transpose: # current shape (B, R, T)
            x_np = x_np.transpose(0, 2, 1)

        return x_np

    def apply(self, x, x_hat):
        x_nps = self.ensure_correct_dim(x)
        x_hat_nps = self.ensure_correct_dim(x_hat)

        pearsons = []
        mads = []
        rmses = []
        for b in range(x.shape[0]):
            x_np = x_nps[b,:,:]
            x_hat_np = x_hat_nps[b,:,:]

            x_sw = self.swfcd.from_fmri(x_np)
            x_hat_sw = self.swfcd.from_fmri(x_hat_np)
            
            x_vec = np.asarray(x_sw["swFCD"], dtype=float).ravel()
            x_hat_vec = np.asarray(x_hat_sw["swFCD"], dtype=float).ravel()

            # calculate metrics:
            common_len = min(x_vec.size, x_hat_vec.size)
            x_vec = x_vec[:common_len]
            x_hat_vec = x_hat_vec[:common_len]

            pearsons.append(np.corrcoef(x_vec, x_hat_vec)[0, 1])
            mads.append(np.mean(np.abs(x_vec - x_hat_vec)))
            rmses.append(np.sqrt(np.mean((x_vec - x_hat_vec) ** 2)))

        return {
            "pearson": pearsons,
            "mad": mads,
            "rmse": rmses
        }
