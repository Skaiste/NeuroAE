from pathlib import Path

import numpy as np

from .. import utils as _utils  # noqa: F401
from DataLoaders.HCP_Schaefer2018 import HCP as LibBrain_HCP


class HCP(LibBrain_HCP):
    def __init__(self, path, parcelations=100):
        self.SchaeferSize = int(parcelations)
        self.set_basePath(path)
        self.timeseries = {}
        self.excluded = {}
        self.__loadFilteredData()

    def set_basePath(self, path):
        path = Path(path)
        self.base_folder = path
        self.fMRI_path = str(
            path / "HCP" / str(self.SchaeferSize) / f"hcp_{{}}_LR_schaefer{self.SchaeferSize}.mat"
        )


def load_hcp(data_dir, parcelations=100):
    data = HCP(data_dir, parcelations=parcelations)
    rows = set(
        np.concatenate(
            [
                [
                    i
                    for i in range(data.timeseries["REST1"][(j, "REST1")].shape[0])
                    if np.any(np.isnan(data.timeseries["REST1"][(j, "REST1")][i]))
                ]
                for j in range(len(data.timeseries["REST1"]))
            ]
        )
    )
    for subject in range(len(data.timeseries["REST1"])):
        data.timeseries["REST1"][(subject, "REST1")] = np.delete(
            data.timeseries["REST1"][(subject, "REST1")],
            list(rows),
            axis=0,
        )

    return data
