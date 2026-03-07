import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..utils.np_utils import encode_labels


def logreg_accuracy_cv(latents, labels, random_seed=42):
    y, valid_mask = encode_labels(labels)
    if y.size == 0:
        return np.nan

    z = latents[valid_mask]
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.nan

    min_count = int(np.min(counts))
    if min_count < 2:
        return np.nan

    n_splits = min(5, min_count)
    clf = LogisticRegression(max_iter=5000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    if len(z.shape) > 2: # if the latent space is 2D
        z = z.reshape(z.shape[0], -1)
    scores = cross_val_score(clf, z, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores))