"""
Data loading functions for ADNI-B dataset using LibBrain DataLoaders.

This module provides convenient functions to load ADNI-B data from the data directory.
The functions wrap the LibBrain/DataLoaders/ADNI_B.py classes and provide a clean
interface for use in main.py and other scripts.
"""

import sys
import csv
import platform
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
from tools.hdf import loadmat


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
        
        self.data = np.array(self.data)

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

        if truncate_features:
            ft_size = self.data.shape[-1] - self.data.shape[-1]%4
            self.data = self.data[:,:,:ft_size]

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


class ADNIDatasetABT(ADNIDataset):
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
        timeseries_data, self.abeta_data, self.tau_data = data
        super().__init__(timeseries_data, labels, subject_ids, filter, normaliser, fit_normaliser, transpose, flatten, pad_features, truncate_features, timepoints_as_samples, fc_input, preserve_timepoints)

    def normalise_abeta_tau(self, abeta_normaliser, tau_normaliser, fit=False):
        """
        Normalises ABeta and Tau levels using z-score
        If values are missing, sets them as 0s
        """
        is_none = lambda n:(n==None).all()
        data_size = max([ab.size for ab in self.abeta_data if not is_none(ab)])
        abeta_data = [np.asarray(ts) if not is_none(ts) else np.asarray([np.nan]*data_size) for ts in self.abeta_data]
        tau_data = [np.asarray(ts) if not is_none(ts) else np.asarray([np.nan]*data_size) for ts in self.tau_data]

        abeta_data = np.array(abeta_data, dtype=np.float32)
        tau_data = np.array(tau_data, dtype=np.float32)

        if fit:
            abeta_data = abeta_normaliser.fit_transform(abeta_data)
            tau_data = tau_normaliser.fit_transform(tau_data)
        else:
            abeta_data = abeta_normaliser.transform(abeta_data)
            tau_data = tau_normaliser.transform(tau_data)

        self.abeta_data = np.nan_to_num(abeta_data)
        self.tau_data = np.nan_to_num(tau_data)

    def __getitem__(self, idx):
        data, label = super().__getitem__(idx)
        return data, (label, self.abeta_data[idx], self.tau_data[idx])


def extract_timeseries_from_loader(data_loader, groups=None):
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
    abeta_list = []
    tau_list = []
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
            abeta = subject_data[subject_id].get("ABeta")
            tau = subject_data[subject_id].get("Tau")
            if timeseries is None:
                print(f"Warning: Missing 'timeseries' for subject {subject_id} in group {group}")
                continue
            
            # Ensure timeseries is numpy array
            if not isinstance(timeseries, np.ndarray):
                timeseries = np.array(timeseries)

            # Ensure abeta is numpy array
            if not isinstance(abeta, np.ndarray):
                abeta = np.array(abeta)
                
            # Ensure tau is numpy array
            if not isinstance(tau, np.ndarray):
                tau = np.array(tau)
            
            timeseries_list.append(timeseries)
            abeta_list.append(abeta)
            tau_list.append(tau)
            subject_ids.append(subject_id)
            labels.append(group)
    
    return timeseries_list, abeta_list, tau_list, subject_ids, labels


def _resolve_split_path(split_path):
    if split_path is None:
        return None
    split_path = Path(split_path)
    if split_path.is_absolute():
        return split_path
    project_root = Path(__file__).resolve().parents[2]
    return project_root / split_path


def _load_split_assignments(split_path):
    id_column = "subject_id"
    label_column = "label"
    split_column = "split"
    assignments = {}
    with split_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {id_column, label_column, split_column}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Split file {split_path} is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            subject_id = str(row[id_column]).strip()
            split_name = str(row[split_column]).strip().lower()
            if split_name not in {"train", "val", "test"}:
                raise ValueError(
                    f"Invalid split '{split_name}' for subject '{subject_id}' in {split_path}."
                )
            assignments[subject_id] = {
                "split": split_name,
                "label": str(row[label_column]).strip(),
            }
    return assignments


def _save_split_assignments(split_path, rows):
    id_column = "subject_id"
    label_column = "label"
    split_column = "split"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[id_column, label_column, split_column],
        )
        writer.writeheader()
        for subject_id, label, split_name in sorted(rows, key=lambda row: (row[2], row[0])):
            writer.writerow(
                {
                    id_column: subject_id,
                    label_column: label,
                    split_column: split_name,
                }
            )


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
    split_mode="none",
    datasplit_file=None,
    preserve_timepoints=False,
    num_workers=0,
    filter=None,
    normaliser=None,
    use_abeta_tau=False
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
    # Extract all timeseries data
    if train_groups is None:
        train_groups = data_loader.get_groupLabels()
    
    all_timeseries, all_abeta, all_tau, all_subject_ids, all_labels = extract_timeseries_from_loader(
        data_loader, groups=train_groups
    )
    
    # Split data if val/test groups not specified
    if val_groups is None and test_groups is None:
        split_mode = (split_mode or "none").lower()
        supported_split_modes = {"none", "load", "create"}
        if split_mode not in supported_split_modes:
            raise ValueError(
                f"Unsupported split_mode '{split_mode}'. Expected one of {sorted(supported_split_modes)}"
            )

        split_path = _resolve_split_path(datasplit_file) if datasplit_file else None
        if split_mode in {"load", "create"} and split_path is None:
            raise ValueError(
                "split_mode requires 'datasplit_file' to be set when not using none mode."
            )

        train_data, train_abeta, train_tau, train_ids, train_labels = [], [], [], [], []
        val_data, val_abeta, val_tau, val_ids, val_labels = [], [], [], [], []
        test_data, test_abeta, test_tau, test_ids, test_labels = [], [], [], [], []

        split_loaded = False
        if split_mode == "load" and split_path.exists():
            split_assignments = _load_split_assignments(
                split_path,
            )
            seen_subjects = set()
            for timeseries, abeta, tau, subject_id, label in zip(all_timeseries, all_abeta, all_tau, all_subject_ids, all_labels):
                assignment = split_assignments.get(subject_id)
                if assignment is None:
                    raise ValueError(
                        f"Subject '{subject_id}' was not found in split file {split_path}."
                    )
                split_name = assignment["split"]
                seen_subjects.add(subject_id)
                if split_name == "train":
                    train_data.append(timeseries)
                    train_abeta.append(abeta)
                    train_tau.append(tau)
                    train_ids.append(subject_id)
                    train_labels.append(label)
                elif split_name == "val":
                    val_data.append(timeseries)
                    val_abeta.append(abeta)
                    val_tau.append(tau)
                    val_ids.append(subject_id)
                    val_labels.append(label)
                else:
                    test_data.append(timeseries)
                    test_abeta.append(abeta)
                    test_tau.append(tau)
                    test_ids.append(subject_id)
                    test_labels.append(label)

            missing_subjects = set(split_assignments.keys()).difference(seen_subjects)
            if missing_subjects:
                print(
                    f"Warning: {len(missing_subjects)} subjects in {split_path} were not found in the loaded dataset."
                )
            print(f"Loaded data split from {split_path}")
            split_loaded = True
        elif split_mode == "load":
            raise FileNotFoundError(f"split_mode=load, but split file does not exist: {split_path}")

        if not split_loaded:
            np.random.seed(random_seed)
            indices = np.random.permutation(len(all_timeseries))

            n_train = int(len(all_timeseries) * train_split)
            n_val = int(len(all_timeseries) * val_split)

            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]

            train_data = [all_timeseries[i] for i in train_indices]
            train_abeta = [all_abeta[i] for i in train_indices]
            train_tau = [all_tau[i] for i in train_indices]
            train_ids = [all_subject_ids[i] for i in train_indices]
            train_labels = [all_labels[i] for i in train_indices]

            val_data = [all_timeseries[i] for i in val_indices]
            val_abeta = [all_abeta[i] for i in val_indices]
            val_tau = [all_tau[i] for i in val_indices]
            val_ids = [all_subject_ids[i] for i in val_indices]
            val_labels = [all_labels[i] for i in val_indices]

            test_data = [all_timeseries[i] for i in test_indices]
            test_abeta = [all_abeta[i] for i in test_indices]
            test_tau = [all_tau[i] for i in test_indices]
            test_ids = [all_subject_ids[i] for i in test_indices]
            test_labels = [all_labels[i] for i in test_indices]

            if split_mode == "create":
                split_rows = []
                for i in train_indices:
                    split_rows.append((all_subject_ids[i], all_labels[i], "train"))
                for i in val_indices:
                    split_rows.append((all_subject_ids[i], all_labels[i], "val"))
                for i in test_indices:
                    split_rows.append((all_subject_ids[i], all_labels[i], "test"))

                _save_split_assignments(
                    split_path,
                    split_rows,
                )
                print(f"Saved data split to {split_path}")
    else:
        if split_mode and (split_mode.lower() != "none"):
            print("Ignoring split_mode/export_datasplit because val_groups/test_groups were explicitly provided.")
        # Use specified groups
        train_data, train_abeta, train_tau, train_ids, train_labels = extract_timeseries_from_loader(
            data_loader, groups=train_groups
        )
        
        val_data = []
        val_labels = []
        if val_groups:
            val_data, val_abeta, val_tau, val_ids, val_labels = extract_timeseries_from_loader(
                data_loader, groups=val_groups
            )
        
        test_data = []
        test_labels = []
        if test_groups:
            test_data, test_abeta, test_tau, test_ids, test_labels = extract_timeseries_from_loader(
                data_loader, groups=test_groups
            )

    DATASET = ADNIDataset
    if use_abeta_tau:
        DATASET = ADNIDatasetABT
        train_data = (train_data, train_abeta, train_tau)
        val_data = (val_data, val_abeta, val_tau)
        test_data = (test_data, test_abeta, test_tau)
    
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

    abeta_norm = None
    tau_norm = None
    if use_abeta_tau:
        abeta_norm = StandardScaler()
        tau_norm = StandardScaler()
        train_dataset.normalise_abeta_tau(abeta_norm, tau_norm, fit=True)

    print("Training dataset")
    train_dataset.describe()

    pin_memory = platform.system().lower() != "darwin"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    input_dim = train_dataset.data[0].shape
    
    result = {
        'train_loader': train_loader,
        'input_dim': input_dim,
        'timepoint_dim': train_dataset.timepoint_dim,
        'preserve_timepoints': preserve_timepoints,
        'num_samples': {'train': len(train_dataset)},
    }
    
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
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        result['val_loader'] = val_loader
        result['num_samples']['val'] = len(val_dataset)
    
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
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        result['test_loader'] = test_loader
        result['num_samples']['test'] = len(test_dataset)
    
    return result
