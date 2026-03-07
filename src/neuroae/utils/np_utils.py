import torch
import numpy as np
import pandas as pd


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


def encode_labels(labels):
    labels = np.asarray(labels)
    valid_mask = pd.notna(labels)
    labels = labels[valid_mask]
    if labels.size == 0:
        return np.array([]), valid_mask

    _, encoded = np.unique(labels.astype(str), return_inverse=True)
    return encoded.astype(int), valid_mask