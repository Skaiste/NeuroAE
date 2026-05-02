"""
Data loading functions for ADNI-B dataset using LibBrain DataLoaders.

This module provides convenient functions to load ADNI-B data from the data directory.
The functions wrap the LibBrain/DataLoaders/ADNI_B.py classes and provide a clean
interface for use in main.py and other scripts.
"""

import sys
import json
import os
import pickle
import platform
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from .utils import *

from DataLoaders.ADNI_B import (
    ADNI_B_N193_no_filt as LibBrain_ADNI_B_N193_no_filt,
    ADNI_B_Alt,
)
from DataLoaders.HCP_Schaefer1000 import HCP as LibBrain_HCP
from tools.hdf import loadmat


class HCP(LibBrain_HCP):
    def __init__(self, path):
        self.SchaeferSize = 100
        self.set_basePath(path)
        self.timeseries = {}
        self.excluded = {}
        self.__loadFilteredData()

    def set_basePath(self, path):
        self.base_folder = path
        self.fMRI_path = str(path / str(self.SchaeferSize) / 'hcp_{}_LR_schaefer100.mat')
        
def load_hcp(data_dir):
    data = HCP(data_dir)
    # filter out nan data
    rows = set(np.concatenate(
               [[i for i in range(data.timeseries['REST1'][(j, 'REST1')].shape[0]) if np.any(np.isnan(data.timeseries['REST1'][(j, 'REST1')][i]))]
                for j in range(len(data.timeseries['REST1']))]))
    for subject in range(len(data.timeseries['REST1'])):
        data.timeseries['REST1'][(subject, 'REST1')] = np.delete(data.timeseries['REST1'][(subject, 'REST1')], list(rows), axis=0)
    
    # reduce dataset to 400
    # subdata = {}
    # for i in range(200):
    #     subdata[(i, 'REST1')] = data.timeseries['REST1'][(1, 'REST1')]
    # data.timeseries['REST1'] = subdata

    return data


def load_ebrains(data_dir=None, tr=2.0):
    return load_ebrains_bold(data_dir=data_dir, tr=tr)


class ADNI_B_N193_no_filt(LibBrain_ADNI_B_N193_no_filt):
    """
    Adapted ADNI-B N193 dataset loader that properly handles custom data directories.
    
    This subclass fixes the path handling issue in the LibBrain version by ensuring
    the path parameter is correctly passed to the parent class. The LibBrain version
    hardcodes path=None when calling super().__init__(), so we override to fix this.
    """
    
    def __init__(self, path=None, discard_AD_ABminus=True, SchaeferSize=400, use_pvc=True):
        """
        Initialize ADNI-B N193 dataset loader.
        
        Args:
            path: Path to data directory. If None, uses get_data_dir().
                  Should point to directory containing 'ADNI-B/N193_no_filt/' subdirectory.
            discard_AD_ABminus: If True, discard AD subjects with ABeta- status.
            SchaeferSize: Parcellation size (400 or 1000).
            use_pvc: If True, use partial volume correction for ABeta and Tau.
        """
        # Handle default path
        if path is None:
            path = get_data_dir()
        
        # Convert Path object to string if needed
        if isinstance(path, Path):
            path = str(path)
        
        # Ensure path ends with '/' for proper concatenation
        if path and not path.endswith('/'):
            path = path + '/'
        
        # Initialize attributes (replicating parent ADNI_B.__init__ logic)
        self.SchaeferSize = SchaeferSize
        self.use_pvc = use_pvc
        self.groups = ['HC','MCI', 'AD']
        
        # Set base path BEFORE calling parent's _loadAllData
        self.set_basePath(path)
        
        # Initialize data structures
        self.timeseries = {}
        self.burdens = {}
        self.meta_information = None
        
        # Load data (this calls the overridden _loadAllData from ADNI_B_N193_no_filt)
        self._loadAllData()
        
        # Apply discard if needed
        if discard_AD_ABminus:
            self.discardSubjects(['116_S_6543','168_S_6754','022_S_6013','126_S_6721'])

        print(self.get_subject_count())


    def set_basePath(self, path):  #, prefiltered_fMRI):
        super().set_basePath(path)
        # adjust the base path to assume the path already contains the required data
        self.base_193_folder = path


def load_adni_n193(
    data_dir=None,
    discard_AD_ABminus=True,
    SchaeferSize=400,
    use_pvc=True,
):
    """
    Load ADNI-B N193 dataset (no filter).
    
    Args:
        data_dir: Path to data directory. If None, uses get_data_dir().
                  Should point to directory containing all the '.mat' files.
        discard_AD_ABminus: If True, discard AD subjects with ABeta- status.
        SchaeferSize: Parcellation size (400 or 1000).
        use_pvc: If True, use partial volume correction for ABeta and Tau.
        
    Returns:
        ADNI_B_N193_no_filt: DataLoader instance for N193 dataset.
    """
    return ADNI_B_N193_no_filt(
        path=data_dir,
        discard_AD_ABminus=discard_AD_ABminus,
        SchaeferSize=SchaeferSize,
        use_pvc=use_pvc,
    )


def load_adni_alt(
    base_loader,
    new_classification,
):
    """
    Load ADNI-B data with alternate classification scheme.
    
    This allows different classification schemes, such as:
    - ['HC', 'AD'] -> all subjects with labels HC and AD
    - ['HC(AB-)', 'HC(AB+)', 'MCI(AB+)', 'AD(AB+)'] -> subjects with ABeta status
    
    Args:
        base_loader: Base DataLoader instance (e.g., from load_adni_n193 or load_adni_n238rev).
        new_classification: List of group labels for the new classification scheme.
        
    Returns:
        ADNI_B_Alt: DataLoader instance with alternate classification.
    """
    return ADNI_B_Alt(
        OrigDataLoader=base_loader,
        new_classification=new_classification,
    )


def get_data_dir():
    """
    Get the default data directory path.
    
    Returns:
        Path object pointing to the data directory.
        Assumes data is in a 'data' directory at the project root.
    """
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    return data_dir


def load_adni(
    data_dir=None,
    discard_AD_ABminus=False,
    use_pvc=True,
    alt_classification=None
):
    """
    Convenience function to load ADNI-B data with common configurations.
    
    Args:
        data_dir: Path to data directory. If None, uses get_data_dir().
        discard_AD_ABminus: If True, discard AD subjects with ABeta- status.
        use_pvc: If True, use partial volume correction for ABeta and Tau.
        alt_classification: Optional list of group labels for alternate classification.
                           If provided, wraps the base loader with ADNI_B_Alt.
        
    Returns:
        DataLoader instance for the requested dataset.
        
    Examples:
        >>> # Load N193 dataset
        >>> loader = load_adni(data_dir='./data')
        >>> 
        >>> # Load with alternate classification
        >>> loader = load_adni(
        ...     data_dir='./data',
        ...     alt_classification=['HC(AB-)', 'HC(AB+)', 'MCI(AB+)', 'AD(AB+)']
        ... )
    """
    if data_dir is None:
        data_dir = get_data_dir()
    
    # Convert Path object to string if needed
    if isinstance(data_dir, Path):
        data_dir = str(data_dir)

    loader = load_adni_n193(
        data_dir=data_dir,
        discard_AD_ABminus=discard_AD_ABminus,
        SchaeferSize=400,
        use_pvc=use_pvc
    )

    # Apply alternate classification if requested
    if alt_classification is not None:
        loader = load_adni_alt(loader, alt_classification)
    
    return loader


class ADNIDataset(Dataset):
    """
    PyTorch Dataset for ADNI timeseries data.
    
    Each sample is a flattened timeseries (N_ROIs * T_timepoints).
    """
    
    def __init__(self, 
        timeseries_data, 
        labels=None, 
        subject_ids=None,
        filter=None,
        normaliser=None,
        fit_normaliser=False,
        transpose=False, 
        flatten=True,
        pad_features=False, 
        truncate_features=False,
        timepoints_as_samples=False,
        fc_input=False,
        preserve_timepoints=False,
    ):
        """
        Initialize ADNI Dataset.
        
        Args:
            timeseries_data: List or array of timeseries data.
                           Each element should be shape (N_ROIs, T_timepoints)
            labels: Optional list of labels for each sample
            flatten: If True, flatten timeseries to 1D (N_ROIs * T_timepoints)
                    If False, keep 2D shape (N_ROIs, T_timepoints)
            normalize: If True, normalize data to [0, 1] range using min-max scaling
            data_min: Minimum value for normalization (if None, computed from data)
            data_max: Maximum value for normalization (if None, computed from data)
        """
        self.flatten = flatten
        self.transpose = transpose
        self.pad_features = pad_features
        self.truncate_features = truncate_features
        self.timepoints_as_samples = timepoints_as_samples
        self.fc_input = fc_input
        self.original_shape = None
        self.subject_ids = subject_ids
        self.filter = filter
        self.normaliser = normaliser

        self.preserve_timepoints = preserve_timepoints
        self.timepoint_dim = None

        assert not (self.truncate_features and self.pad_features), 'Can only choose to pad or truncate features not both.'
        assert not (self.timepoints_as_samples and self.flatten), 'Can only choose to flatten or to treat timepoints as samples, not both.'
        
        processed_timeseries = [np.asarray(ts) for ts in timeseries_data]
        if self.normaliser is not None and len(processed_timeseries) > 0:
            flattened_data = np.asarray([ts.reshape(-1) for ts in processed_timeseries])
            sample_shape = processed_timeseries[0].shape
            if fit_normaliser:
                flattened_data = self.normaliser.fit_transform(flattened_data)
            else:
                try:
                    flattened_data = self.normaliser.transform(flattened_data)
                except NotFittedError as exc:
                    raise ValueError(
                        "Received an unfitted normaliser with fit_normaliser=False. "
                        "Fit on train first and reuse the same normaliser for val/test."
                    ) from exc
            processed_timeseries = [
                flattened_data[i].reshape(sample_shape) for i in range(flattened_data.shape[0])
            ]
        if self.filter is not None:
            processed_timeseries = [self.filter.filter(ts.T).T for ts in processed_timeseries]

        # Convert to numpy arrays and store
        self.data = []
        for ts_array in processed_timeseries:
            
            self.timepoint_dim = ts_array.shape[-1]

            if transpose or timepoints_as_samples:
                ts_array = ts_array.T

            self.original_shape = ts_array.shape
            
            if flatten and not timepoints_as_samples and not fc_input:
                ts_array = ts_array.flatten()

            self.data.append(ts_array)
        
        self.data = np.asarray(self.data, dtype=np.float32)

        if timepoints_as_samples and not fc_input:
            timepoint_dim = self.data.shape[1]
            self.data = self.data.reshape(-1, self.data.shape[-1])
            labels = np.repeat(labels, timepoint_dim).tolist()
            if subject_ids is not None:
                self.subject_ids = np.repeat(subject_ids, timepoint_dim).tolist()

        if fc_input:
            self.timeseries_to_fc(flatten_output=self.flatten)

        if pad_features and self.data.shape[-1]%4 != 0:
            pad_by = 4 - self.data.shape[-1]%4
            self.data = np.pad(self.data, [(0,0),(0,0),(0,pad_by)], mode='constant', constant_values=0)
            self.data = self.data.astype(np.float32, copy=False)

        if truncate_features:
            ft_size = self.data.shape[-1] - self.data.shape[-1]%4
            self.data = self.data[:,:,:ft_size]
            self.data = self.data.astype(np.float32, copy=False)

        self.labels = labels if labels is not None else [None] * len(self.data)
        if self.subject_ids is None:
            self.subject_ids = [None] * len(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.FloatTensor(self.data[idx])
        label = self.labels[idx]
        
        if label is not None:
            return sample, label
        return sample

    def describe(self):
        print(f"Dataset description:")
        print("="*60)
        print(f"\tshape: {self.data.shape}")
        print(f"\tmean: {self.data.mean()}")
        print(f"\tstd: {self.data.std()}")
        print(f"\tmin: {self.data.min()}")
        print(f"\tmax: {self.data.max()}")
        print("="*60)

    def timeseries_to_fc(
        self,
        roi_axis: int = 0,
        fisher: bool = False,
        flatten_output: bool = False,
        upper_triangle: bool = True,
        include_diagonal: bool = False,
        scale_to_unit_interval: bool = False,
    ):
        """
        Convert each sample from timeseries to a functional connectivity matrix.

        This transforms ``self.data`` in place from shape ``(N, R, T)`` (or
        flattened timeseries ``(N, R*T)``) to FC matrices ``(N, R, R)`` and
        optionally flattens to ``(N, R*R)``.

        Args:
            roi_axis: Axis index of ROI dimension for each sample (0 or 1).
                      If ``0``, each sample is interpreted as ``(R, T)``.
                      If ``1``, each sample is interpreted as ``(T, R)``.
            fisher: If True, applies Fisher transform and inverse transform to
                    average-like behavior and numerical stability:
                    ``FC = tanh(atanh(corrcoef))``.
            flatten_output: If True, flattens each FC matrix to 1D.
            upper_triangle: If True, keep only upper-triangular entries.
            include_diagonal: If True and ``upper_triangle=True``, include diagonal.
            scale_to_unit_interval: If True, clip FC to ``[-1, 1]`` and map
                    values to ``[0, 1]`` via ``(x + 1) / 2``.
        """
        if self.timepoints_as_samples:
            raise ValueError(
                "Cannot compute FC when timepoints_as_samples=True because each row is one timepoint."
            )
        if roi_axis not in (0, 1):
            raise ValueError("roi_axis must be 0 or 1.")

        if self.data.ndim == 3:
            ts_data = self.data
        elif self.data.ndim == 2 and self.flatten:
            if self.original_shape is None:
                raise ValueError("original_shape is required to reshape flattened timeseries.")
            expected_features = int(np.prod(self.original_shape))
            if self.data.shape[1] != expected_features:
                raise ValueError(
                    f"Cannot reshape data of shape {self.data.shape} into (-1, {self.original_shape})."
                )
            ts_data = self.data.reshape((-1,) + tuple(self.original_shape))
        else:
            raise ValueError(
                f"Expected timeseries data with ndim=3, or flattened timeseries with ndim=2. Got shape {self.data.shape}."
            )

        fc_mats = []
        for sample in ts_data:
            ts = sample if roi_axis == 0 else sample.T
            fc = np.corrcoef(ts)
            fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            if fisher:
                # Keep values finite for atanh when diagonal or near-perfect correlations appear.
                fc = np.clip(fc, -0.999999, 0.999999)
                fc = np.tanh(np.arctanh(fc))
            if scale_to_unit_interval:
                fc = np.clip(fc, -1.0, 1.0)
                fc = (fc + 1.0) / 2.0
            fc_mats.append(fc.astype(np.float32))

        self.data = np.asarray(fc_mats, dtype=np.float32)
        self.original_shape = self.data.shape[1:]

        if upper_triangle:
            k = 0 if include_diagonal else 1
            tri = np.triu_indices(self.data.shape[1], k=k)
            self.data = self.data[:, tri[0], tri[1]]
            self.flatten = True
            return self.data

        if flatten_output:
            self.data = self.data.reshape(self.data.shape[0], -1)
            self.flatten = True
        else:
            self.flatten = False


class ADNIDatasetBL(ADNIDataset):
    def __init__(self, 
                 data,
                 labels=None,
                 subject_ids=None, 
                 filter=None, 
                 normaliser=None, 
                 fit_normaliser=False, 
                 transpose=False, 
                 flatten=True, 
                 pad_features=False, 
                 truncate_features=False, 
                 timepoints_as_samples=False, 
                 fc_input=False, 
                 preserve_timepoints=False):
        timeseries_data = data[0]
        super().__init__(timeseries_data, labels, subject_ids, filter, normaliser, fit_normaliser, transpose, flatten, pad_features, truncate_features, timepoints_as_samples, fc_input, preserve_timepoints)
        self.bio_data = data[1]

    def prepare(self, label, normaliser=None, fit=False, means=None):
        is_none = lambda n:(n==None).all()

        data = self.bio_data[label]
        data_size = max([ab.size for ab in data if not is_none(ab)])
        data = [np.asarray(ts) if not is_none(ts) else np.asarray([np.nan]*data_size) for ts in data]
        data = np.array(data, dtype=np.float32)

        if normaliser is not None:
            if fit:
                data = normaliser.fit_transform(data)
            else:
                data = normaliser.transform(data)

            self.bio_data[label] = np.nan_to_num(data)
        else:
            if means is None:
                # create means for every region
                means = np.nanmean(data, axis=0)
            self.bio_data[label] = np.where(np.isnan(data), means, data)
            return means

    def __getitem__(self, idx):
        data, label = super().__getitem__(idx)
        return data, (label, {bl:d[idx] for bl,d in self.bio_data.items()})


class CachedDataset(Dataset):
    def __init__(
        self,
        data,
        labels=None,
        subject_ids=None,
        bio_data=None,
        flatten=True,
        fc_input=False,
        preserve_timepoints=False,
        timepoint_dim=None,
        timepoints_as_samples=False,
        transpose=False,
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = labels if labels is not None else [None] * len(self.data)
        self.subject_ids = subject_ids if subject_ids is not None else [None] * len(self.data)
        self.bio_data = {
            key: np.asarray(values, dtype=np.float32) for key, values in (bio_data or {}).items()
        }
        self.flatten = flatten
        self.fc_input = fc_input
        self.timepoints_as_samples = timepoints_as_samples
        self.transpose = transpose
        self.preserve_timepoints = preserve_timepoints
        self.timepoint_dim = timepoint_dim
        self.original_shape = self.data[0].shape if len(self.data) > 0 else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.as_tensor(self.data[idx], dtype=torch.float32)
        label = self.labels[idx]
        if self.bio_data:
            return sample, (label, {bl: values[idx] for bl, values in self.bio_data.items()})
        if label is not None:
            return sample, label
        return sample

    def describe(self):
        print(f"Dataset description:")
        print("="*60)
        print(f"\tshape: {self.data.shape}")
        print(f"\tmean: {self.data.mean()}")
        print(f"\tstd: {self.data.std()}")
        print(f"\tmin: {self.data.min()}")
        print(f"\tmax: {self.data.max()}")
        print("="*60)


def extract_timeseries_from_loader(data_loader, groups=None, bio_levels=[]):
    """
    Extract timeseries data from ADNI DataLoader.
    
    Args:
        data_loader: ADNI DataLoader instance (from load_adni)
        groups: Optional list of groups to include. If None, includes all groups.
        
    Returns:
        timeseries_list: List of timeseries arrays
        subject_ids: List of subject IDs
        labels: List of group labels for each subject
    """
    timeseries_list = []
    bio_level_lists = {bl:[] for bl in bio_levels}
    subject_ids = []
    labels = []
    
    if groups is None:
        groups = data_loader.get_groupLabels()
    
    for group in groups:
        group_subjects = data_loader.get_groupSubjects(group)
        
        for subject_id in group_subjects:
            subject_data = data_loader.get_subjectData(subject_id)
            if subject_id not in subject_data:
                print(f"Warning: Subject {subject_id} not found in get_subjectData for group {group}")
                continue
            timeseries = subject_data[subject_id].get("timeseries")

            if timeseries is None:
                print(f"Warning: Missing 'timeseries' for subject {subject_id} in group {group}")
                continue
            
            # Ensure timeseries is numpy array
            if not isinstance(timeseries, np.ndarray):
                timeseries = np.array(timeseries)

            timeseries_list.append(timeseries)
            subject_ids.append(subject_id)
            labels.append(group)

            for bl in bio_levels:
                bl_values = subject_data[subject_id].get(bl)
                if not isinstance(bl_values, np.ndarray):
                    bl_values = np.array(bl_values)
                bio_level_lists[bl].append(bl_values)
    
    return timeseries_list, bio_level_lists, subject_ids, labels


def _resolve_split_path(split_path):
    if split_path is None:
        return None
    split_path = Path(split_path)
    if split_path.is_absolute():
        return split_path
    project_root = Path(__file__).resolve().parents[2]
    return project_root / split_path


def _build_cache_signature(
    data_type,
    train_groups,
    val_groups,
    test_groups,
    transpose,
    flatten,
    pad_features,
    truncate_features,
    train_split,
    val_split,
    random_seed,
    timepoints_as_samples,
    fc_input,
    preserve_timepoints,
    use_bio_levels,
    normaliser,
    filter,
):
    return {
        "data_type": data_type,
        "train_groups": list(train_groups) if train_groups is not None else None,
        "val_groups": list(val_groups) if val_groups is not None else None,
        "test_groups": list(test_groups) if test_groups is not None else None,
        "transpose": bool(transpose),
        "flatten": bool(flatten),
        "pad_features": bool(pad_features),
        "truncate_features": bool(truncate_features),
        "train_split": float(train_split),
        "val_split": float(val_split),
        "random_seed": int(random_seed),
        "timepoints_as_samples": bool(timepoints_as_samples),
        "fc_input": bool(fc_input),
        "preserve_timepoints": bool(preserve_timepoints),
        "use_bio_levels": list(use_bio_levels),
        "normalizer": (
            {
                "type": type(normaliser).__name__,
                "params": normaliser.get_params(deep=True),
            }
            if normaliser is not None else None
        ),
        "filter": (
            {
                "type": type(filter).__name__,
                "params": {
                    key: value
                    for key, value in vars(filter).items()
                    if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
                },
            }
            if filter is not None else None
        ),
    }


def _save_preprocessed_cache(cache_path, signature, datasets):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "splits": {},
    }
    for split_name, dataset in datasets.items():
        payload["splits"][split_name] = {
            "data": dataset.data.astype(np.float32, copy=False),
            "labels": list(dataset.labels),
            "subject_ids": list(dataset.subject_ids),
            "bio_data": {
                key: np.asarray(values, dtype=np.float32)
                for key, values in getattr(dataset, "bio_data", {}).items()
            },
            "flatten": bool(getattr(dataset, "flatten", False)),
            "fc_input": bool(getattr(dataset, "fc_input", False)),
            "preserve_timepoints": bool(getattr(dataset, "preserve_timepoints", False)),
            "timepoint_dim": getattr(dataset, "timepoint_dim", None),
        }
    temp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        torch.save(payload, temp_cache_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_cache_path, cache_path)
    except Exception:
        if temp_cache_path.exists():
            temp_cache_path.unlink()
        raise


def _load_preprocessed_cache(cache_path, expected_signature=None):
    if cache_path.stat().st_size == 0:
        raise ValueError(
            f"Cache file is empty: {cache_path}. Rebuild it with cache_mode=create."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    # cache_signature = payload.get("signature", {})
    # if expected_signature is not None:
    #     if json.dumps(cache_signature, sort_keys=True) != json.dumps(expected_signature, sort_keys=True):
    #         raise ValueError(
    #             f"Cache at {cache_path} does not match the requested preprocessing configuration."
    #         )

    datasets = {}
    for split_name, split_payload in payload.get("splits", {}).items():
        datasets[split_name] = CachedDataset(
            data=split_payload["data"],
            labels=split_payload.get("labels"),
            subject_ids=split_payload.get("subject_ids"),
            bio_data=split_payload.get("bio_data"),
            flatten=split_payload.get("flatten", False),
            fc_input=split_payload.get("fc_input", False),
            preserve_timepoints=split_payload.get("preserve_timepoints", False),
            timepoint_dim=split_payload.get("timepoint_dim"),
            timepoints_as_samples=split_payload.get("timepoints_as_samples", False),
            transpose=split_payload.get("transpose", False),
        )
    return datasets


def _build_data_loader_result(datasets, batch_size, shuffle_train, num_workers, random_seed, preserve_timepoints):
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(int(random_seed))
    pin_memory = platform.system().lower() != "darwin"

    train_dataset = datasets["train"]
    print("Training dataset")
    train_dataset.describe()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=data_loader_generator,
    )

    result = {
        "train_loader": train_loader,
        "input_dim": train_dataset.data[0].shape,
        "timepoint_dim": train_dataset.timepoint_dim,
        "preserve_timepoints": preserve_timepoints,
        "num_samples": {"train": len(train_dataset)},
    }

    if "val" in datasets:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
        )
        result["val_loader"] = val_loader
        result["num_samples"]["val"] = len(datasets["val"])

    if "test" in datasets:
        test_loader = DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=data_loader_generator,
        )
        result["test_loader"] = test_loader
        result["num_samples"]["test"] = len(datasets["test"])

    return result


def prepare_data_loaders(
    data_loader,
    train_groups=None,
    val_groups=None,
    test_groups=None,
    batch_size=32,
    shuffle_train=True,
    transpose=False,
    flatten=True,
    pad_features=False,
    truncate_features=False,
    train_split=0.7,
    val_split=0.15,
    random_seed=42,
    timepoints_as_samples=False,
    fc_input=False,
    cache_mode="none",
    cache_file=None,
    preserve_timepoints=False,
    num_workers=0,
    filter=None,
    normaliser=None,
    use_bio_levels=[],
    data_type=None,
):
    """
    Prepare PyTorch DataLoaders from ADNI DataLoader.
    
    Args:
        data_loader: ADNI DataLoader instance
        train_groups: List of groups to use for training (e.g., ['HC', 'MCI', 'AD'])
        val_groups: List of groups to use for validation. If None, splits train_groups.
        test_groups: List of groups to use for testing. If None, splits train_groups.
        batch_size: Batch size for DataLoaders
        shuffle_train: Whether to shuffle training data
        flatten: Whether to flatten timeseries to 1D
        train_split: Fraction of data for training (if val/test groups not specified)
        val_split: Fraction of data for validation (if val/test groups not specified)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - 'train_loader': Training DataLoader
            - 'val_loader': Validation DataLoader (if applicable)
            - 'test_loader': Test DataLoader (if applicable)
            - 'input_dim': Input dimension for the model
            - 'num_samples': Dictionary with number of samples per split
    """
    cache_mode = (cache_mode or "none").lower()
    supported_cache_modes = {"none", "load", "create"}
    if cache_mode not in supported_cache_modes:
        raise ValueError(
            f"Unsupported cache_mode '{cache_mode}'. Expected one of {sorted(supported_cache_modes)}"
        )
    cache_path = _resolve_split_path(cache_file) if cache_file else None
    if cache_mode in {"load", "create"} and cache_path is None:
        raise ValueError(
            "cache_mode requires 'cache_file' to be set when not using none mode."
        )
    cache_signature = _build_cache_signature(
        data_type=data_type,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        transpose=transpose,
        flatten=flatten,
        pad_features=pad_features,
        truncate_features=truncate_features,
        train_split=train_split,
        val_split=val_split,
        random_seed=random_seed,
        timepoints_as_samples=timepoints_as_samples,
        fc_input=fc_input,
        preserve_timepoints=preserve_timepoints,
        use_bio_levels=use_bio_levels,
        normaliser=normaliser,
        filter=filter,
    )

    if cache_mode == "load":
        if not cache_path.exists():
            raise FileNotFoundError(f"cache_mode=load, but cache file does not exist: {cache_path}")
        cached_datasets = _load_preprocessed_cache(cache_path, expected_signature=cache_signature)
        print(f"Loaded preprocessed data cache from {cache_path}")
        return _build_data_loader_result(
            cached_datasets,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            num_workers=num_workers,
            random_seed=random_seed,
            preserve_timepoints=preserve_timepoints,
        )

    if data_loader is None:
        raise ValueError("data_loader is required unless cache_mode='load'.")

    # Extract all timeseries data
    if train_groups is None:
        train_groups = data_loader.get_groupLabels()
    
    all_timeseries, all_bio_levels, all_subject_ids, all_labels = extract_timeseries_from_loader(
        data_loader, groups=train_groups, bio_levels=use_bio_levels
    )
    
    # Split data if val/test groups not specified
    if val_groups is None and test_groups is None:
        train_data, train_ids, train_labels = [], [], []
        train_bio_levels = {bl:[] for bl in use_bio_levels}
        val_data, val_ids, val_labels = [], [], []
        val_bio_levels = {bl:[] for bl in use_bio_levels}
        test_data, test_ids, test_labels = [], [], []
        test_bio_levels = {bl:[] for bl in use_bio_levels}
        split_rng = np.random.default_rng(random_seed)
        indices = split_rng.permutation(len(all_timeseries))

        n_train = int(len(all_timeseries) * train_split)
        n_val = int(len(all_timeseries) * val_split)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_data = [all_timeseries[i] for i in train_indices]
        train_ids = [all_subject_ids[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        for bl in use_bio_levels:
            train_bio_levels[bl] = [all_bio_levels[bl][i] for i in train_indices]

        val_data = [all_timeseries[i] for i in val_indices]
        val_ids = [all_subject_ids[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]
        for bl in use_bio_levels:
            val_bio_levels[bl] = [all_bio_levels[bl][i] for i in val_indices]

        test_data = [all_timeseries[i] for i in test_indices]
        test_ids = [all_subject_ids[i] for i in test_indices]
        test_labels = [all_labels[i] for i in test_indices]
        for bl in use_bio_levels:
            test_bio_levels[bl] = [all_bio_levels[bl][i] for i in test_indices]
    else:
        if cache_mode == "create":
            print("Creating cache from explicitly provided train/val/test groups.")
        # Use specified groups
        train_data, train_bio_levels, train_ids, train_labels = extract_timeseries_from_loader(
            data_loader, groups=train_groups, bio_levels=use_bio_levels
        )
        
        val_data = []
        val_labels = []
        if val_groups:
            val_data, val_bio_levels, val_ids, val_labels = extract_timeseries_from_loader(
                data_loader, groups=val_groups, bio_levels=use_bio_levels
            )
        
        test_data = []
        test_labels = []
        if test_groups:
            test_data, test_bio_levels, test_ids, test_labels = extract_timeseries_from_loader(
                data_loader, groups=test_groups, bio_levels=use_bio_levels
            )

    DATASET = ADNIDataset
    if len(use_bio_levels) > 0:
        DATASET = ADNIDatasetBL
        train_data = (train_data, train_bio_levels)
        val_data = (val_data, val_bio_levels)
        test_data = (test_data, test_bio_levels)
    
    # Create PyTorch datasets with normalization
    train_dataset = DATASET(
        train_data, train_labels, train_ids,
        filter=filter,
        normaliser=normaliser,
        fit_normaliser=normaliser is not None,
        transpose=transpose, flatten=flatten, 
        pad_features=pad_features,
        truncate_features=truncate_features,
        timepoints_as_samples=timepoints_as_samples,
        fc_input=fc_input,
        preserve_timepoints=preserve_timepoints
    )

    bio_norm = {bl:StandardScaler() for bl in use_bio_levels}
    bio_means = {}
    for bl in use_bio_levels:
        # train_dataset.prepare(bl, normaliser=bio_norm[bl], fit=True)
        bio_means[bl] = train_dataset.prepare(bl, fit=True)

    datasets = {"train": train_dataset}

    if val_data:
        val_dataset = DATASET(val_data, val_labels, val_ids,
            filter=filter,
            normaliser=train_dataset.normaliser,
            fit_normaliser=False,
            transpose=transpose, flatten=flatten, 
            pad_features=pad_features,
            truncate_features=truncate_features,
            timepoints_as_samples=timepoints_as_samples,
            fc_input=fc_input,
            preserve_timepoints=preserve_timepoints
        )
        for bl in use_bio_levels:
            # val_dataset.prepare(bl, normaliser=bio_norm[bl])
            val_dataset.prepare(bl, means=bio_means[bl])
        datasets["val"] = val_dataset
    
    if test_data:
        test_dataset = DATASET(test_data, test_labels, test_ids,
            filter=filter,
            normaliser=train_dataset.normaliser,
            fit_normaliser=False,
            transpose=transpose, flatten=flatten, 
            pad_features=pad_features,
            truncate_features=truncate_features,
            timepoints_as_samples=timepoints_as_samples,
            fc_input=fc_input,
            preserve_timepoints=preserve_timepoints
        )
        for bl in use_bio_levels:
            # test_dataset.prepare(bl, normaliser=bio_norm[bl])
            test_dataset.prepare(bl, means=bio_means[bl])
        datasets["test"] = test_dataset

    if cache_mode == "create":
        _save_preprocessed_cache(cache_path, cache_signature, datasets)
        print(f"Saved preprocessed data cache to {cache_path}")

    return _build_data_loader_result(
        datasets,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        random_seed=random_seed,
        preserve_timepoints=preserve_timepoints,
    )
