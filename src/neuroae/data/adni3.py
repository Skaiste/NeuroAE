from pathlib import Path

from .. import utils as _utils  # noqa: F401
from DataLoaders.ADNI_B2 import ADNI_B2 as LibBrain_ADNI_B2
from .parcellation import build_parcellation_name, resolve_parcellation_settings


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

_DEFAULT_PARCELLATION_SIZE_BY_TYPE = {
    "Glasser": 360,
    "Schaefer": 400,
}

_MERGED_GROUPS = ["HC", "MCI", "AD"]
_MERGED_GROUP_LABELS = {
    "HC-": "HC",
    "HC+": "HC",
    "MCI+": "MCI",
    "AD+": "AD",
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


def resolve_adni3_parcellation(parcelation=None, parcellation_type=None):
    if parcellation_type is None and parcelation is not None:
        return resolve_parcellation(parcelation)

    parcellation_type, parcelation_size = resolve_parcellation_settings(
        {
            "parcellation_type": parcellation_type,
            "parcelations": parcelation,
        },
        default_size_by_type=_DEFAULT_PARCELLATION_SIZE_BY_TYPE,
    )
    return resolve_parcellation(build_parcellation_name(parcellation_type, parcelation_size))


class ADNI3Loader(LibBrain_ADNI_B2):
    def __init__(
        self,
        path=None,
        parcelation=None,
        parcellation_type=None,
        use_pvc=True,
        merge_groups=True,
    ):
        if path is None:
            path = get_data_dir()

        resolved_parcellation = resolve_adni3_parcellation(
            parcelation=parcelation,
            parcellation_type=parcellation_type,
        )
        self.merge_groups = bool(merge_groups)
        self.SchaeferSize = _PARCELLATION_SIZES[resolved_parcellation]
        super().__init__(
            parcellation=resolved_parcellation,
            path=path,
            use_pvc=use_pvc,
        )
        if self.merge_groups:
            self._merge_loaded_groups()

    def set_basePath(self, path):
        if isinstance(path, Path):
            path = str(path)

        if path and not path.endswith("/"):
            path = path + "/"

        super().set_basePath(path)

    def name(self):
        return "ADNI3"

    def _merge_loaded_groups(self):
        merged_timeseries = {group: {} for group in _MERGED_GROUPS}
        for source_group, target_group in _MERGED_GROUP_LABELS.items():
            merged_timeseries[target_group].update(self.timeseries.get(source_group, {}))
        self.timeseries = merged_timeseries

        if self.burdens:
            merged_burdens = {group: {} for group in _MERGED_GROUPS}
            for source_group, target_group in _MERGED_GROUP_LABELS.items():
                merged_burdens[target_group].update(self.burdens.get(source_group, {}))
            self.burdens = merged_burdens

        self.groups = list(_MERGED_GROUPS)


def load_adni3(
    data_dir=None,
    parcelation=None,
    parcellation_type=None,
    use_pvc=True,
    merge_groups=True,
):
    return ADNI3Loader(
        path=data_dir,
        parcelation=parcelation,
        parcellation_type=parcellation_type,
        use_pvc=use_pvc,
        merge_groups=merge_groups,
    )


__all__ = [
    "ADNI3Loader",
    "get_data_dir",
    "load_adni3",
    "resolve_adni3_parcellation",
    "resolve_parcellation",
]
