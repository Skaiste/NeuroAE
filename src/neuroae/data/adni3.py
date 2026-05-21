from pathlib import Path

from .. import utils as _utils  # noqa: F401
from DataLoaders.ADNI_B2 import ADNI_B2 as LibBrain_ADNI_B2


_PARCELLATION_ALIASES = {
    80: "dbs80",
    100: "Schaefer100",
    360: "Glasser360",
    400: "Schaefer400",
    1000: "Schaefer1000",
    "80": "dbs80",
    "100": "Schaefer100",
    "360": "Glasser360",
    "400": "Schaefer400",
    "1000": "Schaefer1000",
    "dbs80": "dbs80",
    "glasser360": "Glasser360",
    "schaefer100": "Schaefer100",
    "schaefer400": "Schaefer400",
    "schaefer1000": "Schaefer1000",
    "DBS80": "dbs80",
    "Glasser360": "Glasser360",
    "Schaefer100": "Schaefer100",
    "Schaefer400": "Schaefer400",
    "Schaefer1000": "Schaefer1000",
}

_PARCELLATION_SIZES = {
    "dbs80": 80,
    "Glasser360": 360,
    "Schaefer100": 100,
    "Schaefer400": 400,
    "Schaefer1000": 1000,
}


def get_data_dir():
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data"


def resolve_parcellation(parcelation):
    try:
        return _PARCELLATION_ALIASES[parcelation]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ADNI3 parcelation {parcelation!r}. "
            f"Supported values: {sorted(_PARCELLATION_SIZES.values())}"
        ) from exc


class ADNI3Loader(LibBrain_ADNI_B2):
    def __init__(self, path=None, parcelation=400, use_pvc=True):
        if path is None:
            path = get_data_dir()

        resolved_parcellation = resolve_parcellation(parcelation)
        self.SchaeferSize = _PARCELLATION_SIZES[resolved_parcellation]
        super().__init__(
            parcellation=resolved_parcellation,
            path=path,
            use_pvc=use_pvc,
        )

    def set_basePath(self, path):
        if isinstance(path, Path):
            path = str(path)

        if path and not path.endswith("/"):
            path = path + "/"

        super().set_basePath(path)

    def name(self):
        return "ADNI3"


def load_adni3(data_dir=None, parcelation=400, use_pvc=True):
    return ADNI3Loader(
        path=data_dir,
        parcelation=parcelation,
        use_pvc=use_pvc,
    )


__all__ = [
    "ADNI3Loader",
    "get_data_dir",
    "load_adni3",
    "resolve_parcellation",
]
