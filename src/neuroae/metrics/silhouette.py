import numpy as np
from sklearn.metrics import silhouette_score

from ..utils.np_utils import encode_labels



def silhouette(latents, labels):
    if len(latents.shape) > 2: # if the latent space is 2D
        latents = latents.reshape(latents.shape[0], -1)
    y, valid_mask = encode_labels(labels)
    if y.size == 0:
        return np.nan
    z = latents[valid_mask]
    if len(np.unique(y)) < 2 or z.shape[0] <= len(np.unique(y)):
        return np.nan
    return float(silhouette_score(z, y))