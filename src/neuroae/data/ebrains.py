from pathlib import Path

import numpy as np

from .. import utils as _utils  # noqa: F401
from DataLoaders.baseDataLoader import DataLoader as LibBrainDataLoader


class EBRAINSBOLDLoader(LibBrainDataLoader):
    def __init__(
        self,
        base_path,
        *,
        tr=2.25,
        parcelations=100,
        group_label="EBRAINS",
        file_suffix="_RestEmpBOLD.csv",
    ):
        self.base_path = Path(base_path)
        self.tr_seconds = float(tr)
        self.parcelations = int(parcelations)
        self.group_label = group_label
        self.bold_dir = self.base_path / "EBRAINS" / str(self.parcelations) / "BOLD"
        self.file_suffix = file_suffix
        self.timeseries = {}
        self.classification = {}
        self._n_rois = None
        self._load_bold_files()

    def name(self):
        return "EBRAINS"

    def set_basePath(self, path):
        self.base_path = Path(path)
        self.bold_dir = self.base_path / "EBRAINS" / str(self.parcelations) / "BOLD"

    def TR(self):
        return self.tr_seconds

    def N(self):
        return self._n_rois or 0

    def get_classification(self):
        return self.classification

    def get_subjectData(self, subjectID):
        ts = self.timeseries[self.group_label][subjectID]
        return {subjectID: {"timeseries": ts}}

    def discardSubject(self, subjectID):
        if subjectID in self.timeseries.get(self.group_label, {}):
            del self.timeseries[self.group_label][subjectID]
        self.classification.pop(subjectID, None)

    def _load_bold_files(self):
        if not self.bold_dir.exists():
            raise FileNotFoundError(f"EBRAINS BOLD directory does not exist: {self.bold_dir}")

        subject_timeseries = {}
        for csv_path in sorted(self.bold_dir.glob(f"*{self.file_suffix}")):
            subject_id = csv_path.name[: -len(self.file_suffix)]
            timeseries = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
            if timeseries.ndim != 2:
                raise ValueError(
                    f"Expected 2D BOLD signal in {csv_path}, got shape {timeseries.shape}"
                )
            timeseries = timeseries.T
            if not np.isfinite(timeseries).all():
                raise ValueError(f"Found non-finite values in {csv_path}")
            if self._n_rois is None:
                self._n_rois = int(timeseries.shape[0])
            elif timeseries.shape[0] != self._n_rois:
                raise ValueError(
                    f"Inconsistent ROI count in {csv_path}: expected {self._n_rois}, got {timeseries.shape[0]}"
                )
            subject_timeseries[subject_id] = timeseries
            self.classification[subject_id] = self.group_label

        if not subject_timeseries:
            raise FileNotFoundError(
                f"No EBRAINS BOLD files matching '*{self.file_suffix}' were found in {self.bold_dir}"
            )

        self.timeseries[self.group_label] = subject_timeseries


def load_ebrains_bold(data_dir=None, *, tr=2.25, parcelations=100):
    if data_dir is None:
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"
    return EBRAINSBOLDLoader(data_dir, tr=tr, parcelations=parcelations)


def load_ebrains(data_dir=None, tr=2.25, parcelations=100):
    return load_ebrains_bold(data_dir=data_dir, tr=tr, parcelations=parcelations)
