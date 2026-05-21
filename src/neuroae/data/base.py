import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from torch.utils.data import Dataset


class BaseTimeseriesDataset(Dataset):
    def __init__(
        self,
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

        assert not (self.truncate_features and self.pad_features), (
            "Can only choose to pad or truncate features not both."
        )
        assert not (self.timepoints_as_samples and self.flatten), (
            "Can only choose to flatten or to treat timepoints as samples, not both."
        )

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

        if pad_features and self.data.shape[-1] % 4 != 0:
            pad_by = 4 - self.data.shape[-1] % 4
            self.data = np.pad(
                self.data,
                [(0, 0), (0, 0), (0, pad_by)],
                mode="constant",
                constant_values=0,
            )
            self.data = self.data.astype(np.float32, copy=False)

        if truncate_features:
            ft_size = self.data.shape[-1] - self.data.shape[-1] % 4
            self.data = self.data[:, :, :ft_size]
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
        print("Dataset description:")
        print("=" * 60)
        print(f"\tshape: {self.data.shape}")
        print(f"\tmean: {self.data.mean()}")
        print(f"\tstd: {self.data.std()}")
        print(f"\tmin: {self.data.min()}")
        print(f"\tmax: {self.data.max()}")
        print("=" * 60)

    def timeseries_to_fc(
        self,
        roi_axis=0,
        fisher=False,
        flatten_output=False,
        upper_triangle=True,
        include_diagonal=False,
        scale_to_unit_interval=False,
    ):
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
                "Expected timeseries data with ndim=3, or flattened timeseries with ndim=2. "
                f"Got shape {self.data.shape}."
            )

        fc_mats = []
        for sample in ts_data:
            ts = sample if roi_axis == 0 else sample.T
            fc = np.corrcoef(ts)
            fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            if fisher:
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


class BioLevelDataset(BaseTimeseriesDataset):
    def __init__(
        self,
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
        preserve_timepoints=False,
    ):
        timeseries_data = data[0]
        super().__init__(
            timeseries_data,
            labels,
            subject_ids,
            filter,
            normaliser,
            fit_normaliser,
            transpose,
            flatten,
            pad_features,
            truncate_features,
            timepoints_as_samples,
            fc_input,
            preserve_timepoints,
        )
        self.bio_data = data[1]

    def prepare(self, label, normaliser=None, fit=False, means=None):
        is_none = lambda n: (n == None).all()

        data = self.bio_data[label]
        data_size = max([ab.size for ab in data if not is_none(ab)])
        data = [np.asarray(ts) if not is_none(ts) else np.asarray([np.nan] * data_size) for ts in data]
        data = np.array(data, dtype=np.float32)

        if normaliser is not None:
            if fit:
                data = normaliser.fit_transform(data)
            else:
                data = normaliser.transform(data)

            self.bio_data[label] = np.nan_to_num(data)
        else:
            if means is None:
                means = np.nanmean(data, axis=0)
            self.bio_data[label] = np.where(np.isnan(data), means, data)
            return means

    def __getitem__(self, idx):
        data, label = super().__getitem__(idx)
        return data, (label, {bl: d[idx] for bl, d in self.bio_data.items()})


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
        pad_features=False,
        truncate_features=False,
        original_shape=None,
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
        self.pad_features = bool(pad_features)
        self.truncate_features = bool(truncate_features)
        if original_shape is None:
            self.original_shape = self.data[0].shape if len(self.data) > 0 else None
        else:
            self.original_shape = tuple(original_shape)

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
        print("Dataset description:")
        print("=" * 60)
        print(f"\tshape: {self.data.shape}")
        print(f"\tmean: {self.data.mean()}")
        print(f"\tstd: {self.data.std()}")
        print(f"\tmin: {self.data.min()}")
        print(f"\tmax: {self.data.max()}")
        print("=" * 60)
