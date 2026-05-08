from pathlib import Path
import re

import h5py
import numpy as np
import scipy.io as sio

from . import utils as _utils  # noqa: F401
from DataLoaders.baseDataLoader import DataLoader


DEFAULT_GROUP_GLOB_PATTERNS = {
    "AD": ["*ADNI*AD*.mat"],
    "HC_ABetaNeg": ["*HC*ABeta*Neg*.mat", "*HC*ABetaNeg*.mat", "*HC*AB-*neg*.mat"],
    "HC_ABetaPos": ["*HC*ABeta*Pos*.mat", "*HC*ABetaPos*.mat", "*HC*AB+*.mat"],
    "MCI": ["*MCI*ABeta*Pos*.mat", "*MCI*ABetaPos*.mat", "*MCI*AB+*.mat"],
}
DEFAULT_SEPARATE_GROUP_LABELS = {
    "AD": "AD",
    "HC_ABetaNeg": "HC_AB-",
    "HC_ABetaPos": "HC_AB+",
    "MCI": "MCI",
}
DEFAULT_MERGED_GROUP_LABELS = {
    "AD": "AD",
    "HC_ABetaNeg": "HC",
    "HC_ABetaPos": "HC",
    "MCI": "MCI",
}


def get_data_dir():
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data"


def build_relative_data_dir(parcelation):
    return Path("ADNI") / f"Schaefer{int(parcelation)}" / "tseries" / "CONN_denoised_pipeline"


def resolve_adni2_data_dir(data_dir=None, parcelation=100, relative_data_dir=None):
    if data_dir is None:
        data_dir = get_data_dir()

    data_dir = Path(data_dir)
    if relative_data_dir is None:
        relative_data_dir = build_relative_data_dir(parcelation)

    if (data_dir / relative_data_dir).exists():
        return data_dir / relative_data_dir
    return data_dir


def visible_mat_keys(mat_dict):
    return [key for key in mat_dict.keys() if not key.startswith("__")]


def normalise_subject_id(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return normalise_subject_id(value.reshape(-1)[0])
        flattened = value.reshape(-1)
        return "_".join(str(normalise_subject_id(item)) for item in flattened)

    return str(value).strip()


def looks_like_id_array(array):
    if isinstance(array, dict):
        return False

    arr = np.asarray(array)
    if arr.ndim == 0:
        return isinstance(arr.item(), (str, bytes))
    if arr.dtype.kind in {"U", "S"}:
        return True
    if arr.dtype == object:
        flattened = arr.reshape(-1)
        sample = [item for item in flattened[:5] if item is not None]
        if not sample:
            return False
        return all(np.asarray(item).dtype.kind in {"U", "S", "O"} for item in sample)
    return False


def is_hdf5_mat_file(file_path):
    try:
        with h5py.File(file_path, "r"):
            return True
    except OSError:
        return False


def read_hdf5_dataset(file_handle, dataset):
    data = dataset[()]

    if isinstance(data, bytes):
        return data.decode("utf-8")

    if isinstance(data, np.ndarray) and data.dtype.kind in {"S", "U"}:
        return data

    if isinstance(data, np.ndarray) and data.dtype == object:
        refs = data.reshape(-1)
        dereferenced = [read_hdf5_node(file_handle, file_handle[ref]) for ref in refs]
        object_array = np.empty(data.shape, dtype=object)
        for idx, value in enumerate(dereferenced):
            object_array.flat[idx] = value
        return object_array

    return np.asarray(data)


def read_hdf5_group(file_handle, group):
    return {key: read_hdf5_node(file_handle, group[key]) for key in group.keys()}


def read_hdf5_node(file_handle, node):
    if isinstance(node, h5py.Dataset):
        return read_hdf5_dataset(file_handle, node)
    if isinstance(node, h5py.Group):
        return read_hdf5_group(file_handle, node)
    raise TypeError(f"Unsupported HDF5 node type: {type(node)}")


def load_mat_file(file_path):
    if is_hdf5_mat_file(file_path):
        with h5py.File(file_path, "r") as file_handle:
            return {key: read_hdf5_node(file_handle, file_handle[key]) for key in file_handle.keys()}
    return sio.loadmat(file_path)


def normalise_timeseries_shape(timeseries, expected_timepoints=197):
    ts = np.asarray(timeseries)
    if ts.ndim != 2:
        return ts
    if ts.shape[0] == expected_timepoints and ts.shape[1] != expected_timepoints:
        return ts.T
    return ts


def resolve_group_file(data_dir, group_name, override, patterns):
    if override is not None:
        path = Path(override)
        if not path.is_absolute():
            path = data_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Configured file for {group_name} does not exist: {path}")
        return path

    matches = []
    for pattern in patterns:
        matches.extend(sorted(data_dir.glob(pattern)))

    matches = list(dict.fromkeys(matches))
    if not matches:
        raise FileNotFoundError(
            f"Could not auto-discover a MAT file for {group_name} under {data_dir}. "
            f"Tried: {patterns}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Auto-discovery for {group_name} matched multiple files: {matches}. "
            "Set group_file_overrides[...] explicitly."
        )
    return matches[0]


def resolve_timeseries_key(mat_dict, group_name, override=None):
    if override is not None:
        return override

    keys = visible_mat_keys(mat_dict)
    preferred = [
        key for key in keys
        if ("tseries" in key.lower() or "timeseries" in key.lower()) and group_name.lower() in key.lower()
    ]
    if len(preferred) == 1:
        return preferred[0]

    fallback = [key for key in keys if "tseries" in key.lower() or "timeseries" in key.lower()]
    if len(fallback) == 1:
        return fallback[0]

    object_keys = [key for key in keys if np.asarray(mat_dict[key]).dtype == object]
    if len(object_keys) == 1:
        return object_keys[0]

    raise ValueError(
        f"Could not determine the timeseries key for {group_name}. "
        f"Visible keys: {keys}. Set timeseries_key_overrides[{group_name!r}]."
    )


def resolve_id_key(mat_dict, group_name, override=None):
    if override is not None:
        return override

    keys = visible_mat_keys(mat_dict)
    preferred = [
        key for key in keys
        if ("ptid" in key.lower() or re.search(r"\bid\b", key.lower())) and group_name.lower() in key.lower()
    ]
    if len(preferred) == 1:
        return preferred[0]

    fallback = [key for key in keys if "ptid" in key.lower() or "id" in key.lower()]
    if len(fallback) == 1:
        return fallback[0]

    guessed = [key for key in keys if looks_like_id_array(mat_dict[key])]
    if len(guessed) == 1:
        return guessed[0]

    return None


def extract_subject_ids(mat_dict, group_name, n_subjects, override=None):
    id_key = resolve_id_key(mat_dict, group_name, override=override)
    if id_key is None:
        return [f"{group_name}_{idx:03d}" for idx in range(n_subjects)], None

    raw_ids = np.asarray(mat_dict[id_key]).reshape(-1)
    ids = [normalise_subject_id(value) for value in raw_ids]
    if len(ids) != n_subjects:
        raise ValueError(
            f"Subject ID count mismatch for {group_name}: {len(ids)} IDs vs {n_subjects} timeseries entries."
        )
    return ids, id_key


def extract_timeseries_entries(mat_dict, group_name, expected_timepoints=197, override=None):
    ts_key = resolve_timeseries_key(mat_dict, group_name, override=override)
    raw = np.asarray(mat_dict[ts_key]).reshape(-1)
    timeseries = [
        normalise_timeseries_shape(entry, expected_timepoints=expected_timepoints)
        for entry in raw
    ]
    return timeseries, ts_key


class ADNI2Loader(DataLoader):
    def __init__(
        self,
        data_dir=None,
        merge_hc_groups=True,
        parcelation=100,
        relative_data_dir=None,
        group_file_overrides=None,
        group_glob_patterns=None,
        timeseries_key_overrides=None,
        id_key_overrides=None,
        expected_timepoints=197,
        tr=3,
    ):
        self.parcelation = int(parcelation)
        self.data_dir = resolve_adni2_data_dir(
            data_dir,
            parcelation=self.parcelation,
            relative_data_dir=relative_data_dir,
        )
        self.merge_hc_groups = merge_hc_groups
        self.group_file_overrides = group_file_overrides or {}
        self.group_glob_patterns = group_glob_patterns or DEFAULT_GROUP_GLOB_PATTERNS
        self.timeseries_key_overrides = timeseries_key_overrides or {}
        self.id_key_overrides = id_key_overrides or {}
        self.expected_timepoints = expected_timepoints
        self._tr = tr
        self.timeseries = {}
        self.subject_meta = {}
        self.source_files = {}
        self.resolved_keys = {}

        if merge_hc_groups:
            self.groups = ["HC", "MCI", "AD"]
            self.group_labels = DEFAULT_MERGED_GROUP_LABELS
        else:
            self.groups = ["HC_AB-", "HC_AB+", "MCI", "AD"]
            self.group_labels = DEFAULT_SEPARATE_GROUP_LABELS

        for group in self.groups:
            self.timeseries[group] = {}

        self._loadAllData()
        print(self.get_subject_count())

    def name(self):
        return "ADNI2"

    def set_basePath(self, path):
        self.data_dir = Path(path)

    def TR(self):
        return self._tr

    def N(self):
        sample_subject = next(iter(self.get_classification().keys()))
        sample = self.get_subjectData(sample_subject)[sample_subject]["timeseries"]
        return int(np.asarray(sample).shape[0])

    def get_groupLabels(self):
        return self.groups

    def get_classification(self):
        classification = {}
        for group in self.groups:
            for subject_id in self.timeseries[group].keys():
                classification[subject_id] = group
        return classification

    def discardSubject(self, subjectID):
        classification = self.get_classification()
        if subjectID in classification:
            group = classification[subjectID]
            del self.timeseries[group][subjectID]
            self.subject_meta.pop(subjectID, None)

    def get_subjectData(self, subjectID):
        group = self.get_classification()[subjectID]
        return {
            subjectID: {
                "timeseries": self.timeseries[group][subjectID],
                "meta": self.subject_meta[subjectID],
            }
        }

    def _target_group(self, source_group):
        return self.group_labels[source_group]

    def _loadAllData(self):
        source_groups = ["AD", "HC_ABetaNeg", "HC_ABetaPos", "MCI"]

        for source_group in source_groups:
            file_path = resolve_group_file(
                data_dir=self.data_dir,
                group_name=source_group,
                override=self.group_file_overrides.get(source_group),
                patterns=self.group_glob_patterns[source_group],
            )
            self.source_files[source_group] = file_path
            print(f"Loading {file_path}")

            mat = load_mat_file(str(file_path))
            timeseries_entries, ts_key = extract_timeseries_entries(
                mat,
                group_name=source_group,
                expected_timepoints=self.expected_timepoints,
                override=self.timeseries_key_overrides.get(source_group),
            )
            subject_ids, id_key = extract_subject_ids(
                mat,
                group_name=source_group,
                n_subjects=len(timeseries_entries),
                override=self.id_key_overrides.get(source_group),
            )

            self.resolved_keys[source_group] = {
                "timeseries_key": ts_key,
                "id_key": id_key,
            }

            target_group = self._target_group(source_group)
            for subject_id, timeseries in zip(subject_ids, timeseries_entries):
                self.timeseries[target_group][subject_id] = np.asarray(timeseries)
                self.subject_meta[subject_id] = {
                    "source_group": source_group,
                    "loaded_group": target_group,
                    "source_file": str(file_path),
                    "timeseries_key": ts_key,
                    "id_key": id_key,
                }

    def describe_sources(self):
        rows = []
        for source_group, file_path in self.source_files.items():
            target_group = self._target_group(source_group)
            rows.append(
                {
                    "source_group": source_group,
                    "loaded_group": target_group,
                    "file": str(file_path),
                    "timeseries_key": self.resolved_keys[source_group]["timeseries_key"],
                    "id_key": self.resolved_keys[source_group]["id_key"],
                    "subjects_loaded": sum(
                        1
                        for meta in self.subject_meta.values()
                        if meta["source_group"] == source_group
                    ),
                }
            )
        return rows


def load_adni2(
    data_dir=None,
    merge_hc_groups=True,
    parcelation=100,
    relative_data_dir=None,
    group_file_overrides=None,
    group_glob_patterns=None,
    timeseries_key_overrides=None,
    id_key_overrides=None,
    expected_timepoints=197,
    tr=3,
):
    return ADNI2Loader(
        data_dir=data_dir,
        merge_hc_groups=merge_hc_groups,
        parcelation=parcelation,
        relative_data_dir=relative_data_dir,
        group_file_overrides=group_file_overrides,
        group_glob_patterns=group_glob_patterns,
        timeseries_key_overrides=timeseries_key_overrides,
        id_key_overrides=id_key_overrides,
        expected_timepoints=expected_timepoints,
        tr=tr,
    )


__all__ = [
    "ADNI2Loader",
    "build_relative_data_dir",
    "get_data_dir",
    "load_adni2",
    "resolve_adni2_data_dir",
]
