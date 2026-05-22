from pathlib import Path


DEFAULT_PARCELLATION_TYPE = "Schaefer"
DEFAULT_PARCELLATION_SIZE_BY_TYPE = {
    "Schaefer": 100,
    "Glasser": 360,
}

_PARCELLATION_TYPE_ALIASES = {
    "glasser": "Glasser",
    "Glasser": "Glasser",
    "schaefer": "Schaefer",
    "Schaefer": "Schaefer",
}


def resolve_parcellation_type(parcellation_type=None):
    if parcellation_type is None:
        return DEFAULT_PARCELLATION_TYPE

    try:
        return _PARCELLATION_TYPE_ALIASES[str(parcellation_type).strip()]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_PARCELLATION_SIZE_BY_TYPE))
        raise ValueError(
            f"Unsupported parcellation type {parcellation_type!r}. Supported values: {supported}"
        ) from exc


def default_parcellation_size(parcellation_type=None, default_size_by_type=None):
    resolved_type = resolve_parcellation_type(parcellation_type)
    size_by_type = default_size_by_type or DEFAULT_PARCELLATION_SIZE_BY_TYPE
    return size_by_type[resolved_type]


def build_parcellation_name(parcellation_type, parcelations):
    resolved_type = resolve_parcellation_type(parcellation_type)
    return f"{resolved_type}{int(parcelations)}"


def build_parcellation_basename(parcellation_type, parcelations):
    resolved_type = resolve_parcellation_type(parcellation_type)
    return f"{resolved_type.lower()}{int(parcelations)}"


def candidate_parcellation_dirnames(parcellation_type, parcelations):
    canonical_name = build_parcellation_name(parcellation_type, parcelations)
    basename = build_parcellation_basename(parcellation_type, parcelations)
    dirnames = [canonical_name, basename, str(int(parcelations))]
    return list(dict.fromkeys(dirnames))


def resolve_parcellation_settings(data_config=None, default_size_by_type=None):
    config = data_config or {}
    data_section = config.get("data", config)

    parcellation_type = resolve_parcellation_type(
        data_section.get("parcellation_type", data_section.get("parcelation_type"))
    )

    parcelations = data_section.get("parcelations")
    if parcelations is None:
        parcelations = data_section.get("parcellation_size", data_section.get("parcelation_size"))
    if parcelations is None:
        parcelations = default_parcellation_size(
            parcellation_type,
            default_size_by_type=default_size_by_type,
        )

    return parcellation_type, int(parcelations)


def resolve_existing_path(candidates):
    candidate_paths = [Path(candidate) for candidate in candidates]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return candidate_paths[0]
