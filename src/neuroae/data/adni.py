from pathlib import Path

from .. import utils as _utils  # noqa: F401
from DataLoaders.ADNI_B import (
    ADNI_B_Alt,
    ADNI_B_N193_no_filt as LibBrain_ADNI_B_N193_no_filt,
)


class ADNI_B_N193_no_filt(LibBrain_ADNI_B_N193_no_filt):
    def __init__(self, path=None, discard_AD_ABminus=True, SchaeferSize=400, use_pvc=True):
        if path is None:
            path = get_data_dir()

        if isinstance(path, Path):
            path = str(path)

        if path and not path.endswith("/"):
            path = path + "/"

        self.SchaeferSize = SchaeferSize
        self.use_pvc = use_pvc
        self.groups = ["HC", "MCI", "AD"]
        self.set_basePath(path)
        self.timeseries = {}
        self.burdens = {}
        self.meta_information = None
        self._loadAllData()

        if discard_AD_ABminus:
            self.discardSubjects(["116_S_6543", "168_S_6754", "022_S_6013", "126_S_6721"])

        print(self.get_subject_count())

    def set_basePath(self, path):
        super().set_basePath(path)
        self.base_193_folder = path


def get_data_dir():
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data"


def load_adni_n193(
    data_dir=None,
    discard_AD_ABminus=True,
    SchaeferSize=400,
    use_pvc=True,
):
    return ADNI_B_N193_no_filt(
        path=data_dir,
        discard_AD_ABminus=discard_AD_ABminus,
        SchaeferSize=SchaeferSize,
        use_pvc=use_pvc,
    )


def load_adni_alt(base_loader, new_classification):
    return ADNI_B_Alt(
        OrigDataLoader=base_loader,
        new_classification=new_classification,
    )


def load_adni(
    data_dir=None,
    discard_AD_ABminus=False,
    use_pvc=True,
    alt_classification=None,
):
    if data_dir is None:
        data_dir = get_data_dir()

    if isinstance(data_dir, Path):
        data_dir = str(data_dir)

    loader = load_adni_n193(
        data_dir=data_dir,
        discard_AD_ABminus=discard_AD_ABminus,
        SchaeferSize=400,
        use_pvc=use_pvc,
    )

    if alt_classification is not None:
        loader = load_adni_alt(loader, alt_classification)

    return loader
