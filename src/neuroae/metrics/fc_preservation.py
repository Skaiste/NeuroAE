import pandas as pd
import numpy as np

from ..utils.np_utils import to_numpy



def _reshape_for_timeseries(sample, dataset):
    """Try to recover (R, T) for FC-preservation computation."""
    x = to_numpy(sample)

    if getattr(dataset, "fc_input", False):
        return None

    if x.ndim == 2:
        return x

    if x.ndim == 1 and getattr(dataset, "flatten", False):
        original_shape = getattr(dataset, "original_shape", None)
        if original_shape is None:
            return None
        expected = int(np.prod(original_shape))
        if x.size != expected:
            return None
        return x.reshape(original_shape)

    return None


def _fc_upper_vector_from_timeseries(ts, roi_axis=0):
    ts = to_numpy(ts)
    if ts.ndim != 2:
        return None
    if roi_axis == 1:
        ts = ts.T

    fc = np.corrcoef(ts)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    tri = np.triu_indices(fc.shape[0], k=1)
    return fc[tri]


def _fc_vector(sample, dataset):
    ts = _reshape_for_timeseries(sample, dataset)
    if ts is None:
        # fallback: treat sample as already FC-like feature vector
        return to_numpy(sample).reshape(-1)

    # dataset.transpose=True means sample orientation is (T, R), so ROI axis is 1.
    roi_axis = 1 if getattr(dataset, "transpose", False) else 0
    fc_vec = _fc_upper_vector_from_timeseries(ts, roi_axis=roi_axis)
    if fc_vec is None:
        return to_numpy(sample).reshape(-1)
    return fc_vec


def _vector_correlation(a, b):
    a = to_numpy(a).reshape(-1)
    b = to_numpy(b).reshape(-1)

    if a.size != b.size or a.size < 2:
        return np.nan

    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std == 0.0 or b_std == 0.0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def fc_preservation_score(x, x_hat, dataset):
    x_np = to_numpy(x)
    x_hat_np = to_numpy(x_hat)

    if getattr(dataset, "timepoints_as_samples", False):
        subject_ids = np.asarray(getattr(dataset, "subject_ids", []))
        if subject_ids.size != x_np.shape[0]:
            return np.nan

        scores = []
        unique_subjects = pd.unique(subject_ids)
        for sid in unique_subjects:
            idx = np.where(subject_ids == sid)[0]
            if idx.size < 2:
                continue

            # In timepoints_as_samples mode each row is one timepoint vector of ROIs.
            ts_x = x_np[idx]
            ts_hat = x_hat_np[idx]
            v1 = _fc_upper_vector_from_timeseries(ts_x, roi_axis=1)
            v2 = _fc_upper_vector_from_timeseries(ts_hat, roi_axis=1)
            if v1 is None or v2 is None:
                continue

            corr = _vector_correlation(v1, v2)
            if np.isfinite(corr):
                scores.append(corr)
        return float(np.mean(scores)) if scores else np.nan

    scores = []
    for i in range(x_np.shape[0]):
        v1 = _fc_vector(x_np[i], dataset)
        v2 = _fc_vector(x_hat_np[i], dataset)
        corr = _vector_correlation(v1, v2)
        if np.isfinite(corr):
            scores.append(corr)

    return float(np.mean(scores)) if scores else np.nan