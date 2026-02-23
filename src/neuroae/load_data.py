"""
Data loading functions for ADNI-B dataset using LibBrain DataLoaders.

This module provides convenient functions to load ADNI-B data from the data directory.
The functions wrap the LibBrain/DataLoaders/ADNI_B.py classes and provide a clean
interface for use in main.py and other scripts.
"""

import sys
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
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
        self.base_238_folder = path
        self.fMRI_path = str(Path(path) / Path(self.fMRI_path).name)
        self.ID_path = str(Path(path) / Path(self.ID_path).name)

    def _loadAllData(self):
        """
        Override to call LibBrain_ADNI_B_N193_no_filt's fMRI/burden loading logic
        but skip ADNI_B's metadata loading.
        """
        # Copy the fMRI/burden loading logic from LibBrain_ADNI_B_N193_no_filt._loadAllData()
        # but skip the super()._loadAllData() call that would call ADNI_B._loadAllData()
        for task in self.groups:
            print(f'----------- Checking: {task} --------------')
            taskRealName = task
            taskBatch = '123' if task == 'AD' or task == 'HC' else '1'
            ID_path = self.ID_path.format(taskRealName, taskBatch)
            fMRI_task_path = self.fMRI_path.format(taskRealName, taskBatch)
            PTIDs = loadmat(ID_path)
            PTIDs_name = f'combined_PTIDS_ADNI3_{task}_MPRAGE' if task == 'HC' or task == 'AD' \
                else f'PTID_BIDS_MPRAGE_60_89_batch_1_{task}'
            PTIDs = PTIDs[PTIDs_name]
            IDs = [id[0] for id in np.squeeze(PTIDs).tolist()]
            self.timeseries[task] = self._loadSubjects_fMRI(IDs, fMRI_task_path, task)
            self.burdens[task] = self._loadSubjects_burden(IDs)
            print(f'----------- done {task} --------------')
        print(f'----------- done loading All --------------')


class ADNI_B_N193_filtered(ADNI_B_N193_no_filt):
    def __init__(self, filter, path=None, discard_AD_ABminus=True, SchaeferSize=400, use_pvc=True):
        self.filter = filter
        super().__init__(path=path, discard_AD_ABminus=discard_AD_ABminus, SchaeferSize=SchaeferSize, use_pvc=use_pvc)

    def _loadAllData(self):
        super()._loadAllData()
        # apply filtering to the data
        print(f'----------- Filtering data --------------')
        for task in self.groups:
            for subject, data in self.timeseries[task].items():
                self.timeseries[task][subject] = self.filter.filter(data.T).T
            print(f'----------- done filtering for {task} --------------')
        print(f'----------- done filtering All --------------')




def load_adni_n193(
    data_dir=None,
    discard_AD_ABminus=True,
    SchaeferSize=400,
    use_pvc=True,
    filter=None,
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
        ADNI_B_N193_filtered: DataLoader instance for N193 dataset.
    """
    if filter is None:
        return ADNI_B_N193_no_filt(
            path=data_dir,
            discard_AD_ABminus=discard_AD_ABminus,
            SchaeferSize=SchaeferSize,
            use_pvc=use_pvc,
        )
    else:
        return ADNI_B_N193_filtered(
            filter,
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
    discard_AD_ABminus=True,
    use_pvc=True,
    alt_classification=None,
    filter=None,
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
        use_pvc=use_pvc,
        filter=filter,
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
        transpose=False, 
        flatten=True, 
        normaliser=None, 
        pad_features=False, 
        truncate_features=False,
        timepoints_as_samples=False,
        fc_input=False
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
        self.normaliser = normaliser
        self.pad_features = pad_features
        self.truncate_features = truncate_features
        self.timepoints_as_samples = timepoints_as_samples
        self.fc_input = fc_input
        self.original_shape = None

        assert not (self.truncate_features and self.pad_features), 'Can only choose to pad or truncate features not both.'
        assert not (self.timepoints_as_samples and self.flatten), 'Can only choose to flatten or to treat timepoints as samples, not both.'
        
        # Convert to numpy arrays and store
        self.data = []
        for ts in timeseries_data:
            ts_array = np.array(ts)
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

        if normaliser is not None:
            if fc_input or not flatten and not timepoints_as_samples:
                # shape it for normaliser
                N, P, T = self.data.shape
                data_reshaped = self.data.transpose(0, 2, 1)
                data_flat = data_reshaped.reshape(-1, P)
                self.data = data_flat
            # assuming the normaliser will first be applied to training data, 
            # therefore it can be fit on the first usage of the normaliser
            if not hasattr(normaliser, 'mean_'):
                normaliser.fit(self.data)
            self.data = normaliser.transform(self.data)

            if fc_input or not flatten and not timepoints_as_samples:
                # reshape back
                data_scaled = self.data.reshape(N, T, P)
                self.data = data_scaled.transpose(0, 2, 1)

        if fc_input:
            self.timeseries_to_fc(flatten_output=self.flatten)

        if pad_features and self.data.shape[-1]%4 != 0:
            pad_by = 4 - self.data.shape[-1]%4
            self.data = np.pad(self.data, [(0,0),(0,0),(0,pad_by)], mode='constant', constant_values=0)

        if truncate_features:
            ft_size = self.data.shape[-1] - self.data.shape[-1]%4
            self.data = self.data[:,:,:ft_size]

        self.labels = labels if labels is not None else [None] * len(self.data)
        
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
    subject_ids = []
    labels = []
    
    if groups is None:
        groups = data_loader.get_groupLabels()
    
    classification = data_loader.get_classification()
    
    for group in groups:
        group_subjects = data_loader.get_groupSubjects(group)
        
        for subject_id in group_subjects:
            # Access timeseries directly from data_loader.timeseries instead of using get_subjectData
            # This avoids the need for meta_information which may not exist
            # First try the group we're iterating over
            if group in data_loader.timeseries and subject_id in data_loader.timeseries[group]:
                timeseries = data_loader.timeseries[group][subject_id]
            else:
                # Fallback: try to get from classification if timeseries structure is different
                # (e.g., for ADNI_B_Alt which may have different group names)
                group_from_classification = classification.get(subject_id)
                if group_from_classification and group_from_classification in data_loader.timeseries:
                    if subject_id in data_loader.timeseries[group_from_classification]:
                        timeseries = data_loader.timeseries[group_from_classification][subject_id]
                    else:
                        print(f"Warning: Subject {subject_id} not found in timeseries for group {group_from_classification}")
                        continue
                else:
                    print(f"Warning: Subject {subject_id} not found in timeseries for group {group}")
                    continue
            
            # Ensure timeseries is numpy array
            if not isinstance(timeseries, np.ndarray):
                timeseries = np.array(timeseries)
            
            timeseries_list.append(timeseries)
            subject_ids.append(subject_id)
            labels.append(group)
    
    return timeseries_list, subject_ids, labels


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
    normalize=True,
    timepoints_as_samples=False,
    fc_input=False,
    split_mode="none",
    datasplit_file=None,
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
    
    all_timeseries, all_subject_ids, all_labels = extract_timeseries_from_loader(
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

        train_data, train_labels = [], []
        val_data, val_labels = [], []
        test_data, test_labels = [], []

        split_loaded = False
        if split_mode == "load" and split_path.exists():
            split_assignments = _load_split_assignments(
                split_path,
            )
            seen_subjects = set()
            for timeseries, subject_id, label in zip(all_timeseries, all_subject_ids, all_labels):
                assignment = split_assignments.get(subject_id)
                if assignment is None:
                    raise ValueError(
                        f"Subject '{subject_id}' was not found in split file {split_path}."
                    )
                split_name = assignment["split"]
                seen_subjects.add(subject_id)
                if split_name == "train":
                    train_data.append(timeseries)
                    train_labels.append(label)
                elif split_name == "val":
                    val_data.append(timeseries)
                    val_labels.append(label)
                else:
                    test_data.append(timeseries)
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
            train_labels = [all_labels[i] for i in train_indices]

            val_data = [all_timeseries[i] for i in val_indices]
            val_labels = [all_labels[i] for i in val_indices]

            test_data = [all_timeseries[i] for i in test_indices]
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
        train_data, train_ids, train_labels = extract_timeseries_from_loader(
            data_loader, groups=train_groups
        )
        
        val_data = []
        val_labels = []
        if val_groups:
            val_data, val_ids, val_labels = extract_timeseries_from_loader(
                data_loader, groups=val_groups
            )
        
        test_data = []
        test_labels = []
        if test_groups:
            test_data, test_ids, test_labels = extract_timeseries_from_loader(
                data_loader, groups=test_groups
            )

    normaliser = StandardScaler() if normalize else None
    
    # Create PyTorch datasets with normalization
    train_dataset = ADNIDataset(
        train_data, train_labels, 
        transpose=transpose, flatten=flatten, 
        normaliser=normaliser, pad_features=pad_features,
        truncate_features=truncate_features,
        timepoints_as_samples=timepoints_as_samples,
        fc_input=fc_input
    )
    print("Training dataset")
    train_dataset.describe()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )

    input_dim = train_dataset.data[0].shape
    
    result = {
        'train_loader': train_loader,
        'input_dim': input_dim,
        'num_samples': {'train': len(train_dataset)},
    }
    
    if val_data:
        val_dataset = ADNIDataset(val_data, val_labels,
            transpose=transpose, flatten=flatten, 
            normaliser=normaliser, pad_features=pad_features,
            truncate_features=truncate_features,
            timepoints_as_samples=timepoints_as_samples,
            fc_input=fc_input
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        result['val_loader'] = val_loader
        result['num_samples']['val'] = len(val_dataset)
    
    if test_data:
        test_dataset = ADNIDataset(test_data, test_labels,
            transpose=transpose, flatten=flatten, 
            normaliser=normaliser, pad_features=pad_features,
            truncate_features=truncate_features,
            timepoints_as_samples=timepoints_as_samples,
            fc_input=fc_input
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        result['test_loader'] = test_loader
        result['num_samples']['test'] = len(test_dataset)
    
    return result
