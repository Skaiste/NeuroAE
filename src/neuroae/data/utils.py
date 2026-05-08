import json
import os
import pickle
import platform
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .base import BaseTimeseriesDataset, BioLevelDataset, CachedDataset


def extract_timeseries_from_loader(data_loader, groups=None, bio_levels=None):
    timeseries_list = []
    bio_levels = bio_levels or []
    bio_level_lists = {bl: [] for bl in bio_levels}
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


def resolve_split_path(split_path):
    if split_path is None:
        return None
    split_path = Path(split_path)
    if split_path.is_absolute():
        return split_path
    project_root = Path(__file__).resolve().parents[3]
    return project_root / split_path


def build_cache_signature(
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
            if normaliser is not None
            else None
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
            if filter is not None
            else None
        ),
    }


def save_preprocessed_cache(cache_path, signature, datasets):
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


def load_preprocessed_cache(cache_path, expected_signature=None):
    if cache_path.stat().st_size == 0:
        raise ValueError(
            f"Cache file is empty: {cache_path}. Rebuild it with cache_mode=create."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)

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


def build_data_loader_result(
    datasets,
    batch_size,
    shuffle_train,
    num_workers,
    random_seed,
    preserve_timepoints,
):
    def seed_worker(worker_id):
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
        worker_init_fn=seed_worker,
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
            worker_init_fn=seed_worker,
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
            worker_init_fn=seed_worker,
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
    use_bio_levels=None,
    data_type=None,
):
    use_bio_levels = use_bio_levels or []
    cache_mode = (cache_mode or "none").lower()
    supported_cache_modes = {"none", "load", "create"}
    if cache_mode not in supported_cache_modes:
        raise ValueError(
            f"Unsupported cache_mode '{cache_mode}'. Expected one of {sorted(supported_cache_modes)}"
        )
    cache_path = resolve_split_path(cache_file) if cache_file else None
    if cache_mode in {"load", "create"} and cache_path is None:
        raise ValueError("cache_mode requires 'cache_file' to be set when not using none mode.")
    cache_signature = build_cache_signature(
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
        cached_datasets = load_preprocessed_cache(cache_path, expected_signature=cache_signature)
        print(f"Loaded preprocessed data cache from {cache_path}")
        return build_data_loader_result(
            cached_datasets,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            num_workers=num_workers,
            random_seed=random_seed,
            preserve_timepoints=preserve_timepoints,
        )

    if data_loader is None:
        raise ValueError("data_loader is required unless cache_mode='load'.")

    if train_groups is None:
        train_groups = data_loader.get_groupLabels()

    all_timeseries, all_bio_levels, all_subject_ids, all_labels = extract_timeseries_from_loader(
        data_loader,
        groups=train_groups,
        bio_levels=use_bio_levels,
    )

    if val_groups is None and test_groups is None:
        train_bio_levels = {bl: [] for bl in use_bio_levels}
        val_bio_levels = {bl: [] for bl in use_bio_levels}
        test_bio_levels = {bl: [] for bl in use_bio_levels}
        split_rng = np.random.default_rng(random_seed)
        indices = split_rng.permutation(len(all_timeseries))

        n_train = int(len(all_timeseries) * train_split)
        n_val = int(len(all_timeseries) * val_split)

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

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

        train_data, train_bio_levels, train_ids, train_labels = extract_timeseries_from_loader(
            data_loader,
            groups=train_groups,
            bio_levels=use_bio_levels,
        )

        val_data = []
        val_labels = []
        val_ids = []
        val_bio_levels = {bl: [] for bl in use_bio_levels}
        if val_groups:
            val_data, val_bio_levels, val_ids, val_labels = extract_timeseries_from_loader(
                data_loader,
                groups=val_groups,
                bio_levels=use_bio_levels,
            )

        test_data = []
        test_labels = []
        test_ids = []
        test_bio_levels = {bl: [] for bl in use_bio_levels}
        if test_groups:
            test_data, test_bio_levels, test_ids, test_labels = extract_timeseries_from_loader(
                data_loader,
                groups=test_groups,
                bio_levels=use_bio_levels,
            )

    dataset_cls = BaseTimeseriesDataset
    if len(use_bio_levels) > 0:
        dataset_cls = BioLevelDataset
        train_data = (train_data, train_bio_levels)
        val_data = (val_data, val_bio_levels)
        test_data = (test_data, test_bio_levels)

    train_dataset = dataset_cls(
        train_data,
        train_labels,
        train_ids,
        filter=filter,
        normaliser=normaliser,
        fit_normaliser=normaliser is not None,
        transpose=transpose,
        flatten=flatten,
        pad_features=pad_features,
        truncate_features=truncate_features,
        timepoints_as_samples=timepoints_as_samples,
        fc_input=fc_input,
        preserve_timepoints=preserve_timepoints,
    )

    bio_norm = {bl: StandardScaler() for bl in use_bio_levels}
    bio_means = {}
    for bl in use_bio_levels:
        bio_means[bl] = train_dataset.prepare(bl, fit=True)

    datasets = {"train": train_dataset}

    if val_data:
        val_dataset = dataset_cls(
            val_data,
            val_labels,
            val_ids,
            filter=filter,
            normaliser=train_dataset.normaliser,
            fit_normaliser=False,
            transpose=transpose,
            flatten=flatten,
            pad_features=pad_features,
            truncate_features=truncate_features,
            timepoints_as_samples=timepoints_as_samples,
            fc_input=fc_input,
            preserve_timepoints=preserve_timepoints,
        )
        for bl in use_bio_levels:
            val_dataset.prepare(bl, means=bio_means[bl])
        datasets["val"] = val_dataset

    if test_data:
        test_dataset = dataset_cls(
            test_data,
            test_labels,
            test_ids,
            filter=filter,
            normaliser=train_dataset.normaliser,
            fit_normaliser=False,
            transpose=transpose,
            flatten=flatten,
            pad_features=pad_features,
            truncate_features=truncate_features,
            timepoints_as_samples=timepoints_as_samples,
            fc_input=fc_input,
            preserve_timepoints=preserve_timepoints,
        )
        for bl in use_bio_levels:
            test_dataset.prepare(bl, means=bio_means[bl])
        datasets["test"] = test_dataset

    if cache_mode == "create":
        save_preprocessed_cache(cache_path, cache_signature, datasets)
        print(f"Saved preprocessed data cache to {cache_path}")

    return build_data_loader_result(
        datasets,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        random_seed=random_seed,
        preserve_timepoints=preserve_timepoints,
    )
