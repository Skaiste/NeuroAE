"""ADNI-Long progression-pair loader matching the longitudinal notebook."""

from __future__ import annotations

import numpy as np

from DataLoaders.ADNI_Long import ADNI_Long as LibBrainADNILong

from ..filters import NilearnBandPassFilter
from .adni3 import get_data_dir


DEFAULT_ALLOWED_PROGRESSIONS = frozenset({("HC", "MCI"), ("HC", "AD"), ("MCI", "AD")})


def _normalise_diagnosis(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return None
    return "HC" if value == "CN" else value


def progression_pairs(diagnoses, allowed_progressions=DEFAULT_ALLOWED_PROGRESSIONS):
    """Return chronological indices whose diagnoses form an allowed progression."""
    diagnoses = [_normalise_diagnosis(value) for value in diagnoses]
    return [
        (initial_idx, progressed_idx)
        for initial_idx, initial_diagnosis in enumerate(diagnoses)
        for progressed_idx, progressed_diagnosis in enumerate(
            diagnoses[initial_idx + 1:], initial_idx + 1
        )
        if (initial_diagnosis, progressed_diagnosis) in allowed_progressions
    ]


def has_allowed_progression(diagnoses, allowed_progressions=DEFAULT_ALLOWED_PROGRESSIONS):
    """Return true when a chronological pair forms an allowed progression."""
    return bool(progression_pairs(diagnoses, allowed_progressions))


class ADNILongLoader(LibBrainADNILong):
    """Load and preprocess ADNI-Long diagnosed progression pairs.

    Samples are ``sub-.../initial-session/progressed-session`` pairs. The data
    array is ``(2, timepoints, regions)``: initial then progressed session.
    Every chronological HC→MCI, HC→AD, and MCI→AD combination is retained.
    """

    def __init__(
        self, path=None, parcellation="Schaefer400", template_name="template_all",
        fmri_deriv_name="fmri_prepro_denoised_bp_008_08", prefer_cl=True,
        prefer_pvc=True, allowed_progressions=DEFAULT_ALLOWED_PROGRESSIONS,
        filter_timeseries=True, high_pass=0.008, low_pass=0.08, detrend=False,
        normalise=True, strict=False, verbose=False,
    ):
        self.allowed_progressions = frozenset(allowed_progressions)
        self.filter_timeseries = bool(filter_timeseries)
        self.normalise = bool(normalise)
        self._pair_samples = {}
        self._pair_classification = {}
        self._progression_subjects = []
        self.min_timepoints = None
        super().__init__(
            path=str(get_data_dir() if path is None else path), parcellation=parcellation,
            template_name=template_name, fmri_deriv_name=fmri_deriv_name,
            prefer_cl=prefer_cl, prefer_pvc=prefer_pvc, load_data=True,
            strict=strict, verbose=False,
        )
        self.nilearn_filter = (
            NilearnBandPassFilter(self.TR(), high_pass, low_pass, detrend)
            if self.filter_timeseries else None
        )
        self._build_progression_pairs()
        if verbose:
            print(self.summary())

    def name(self):
        return "ADNI_Long"

    def get_classification(self):
        return dict(self._pair_classification)

    def get_subject_count(self):
        """Return progression-pair counts per progressed diagnosis."""
        return {
            group: len(self.get_groupSubjects(group))
            for group in self.get_groupLabels()
        }

    def get_subjectData(self, subjectID):
        if subjectID not in self._pair_samples:
            raise KeyError(f"Unknown ADNI-Long progression pair: {subjectID}")
        return {subjectID: self._pair_samples[subjectID]}

    def get_subjects(self):
        """Return progression-pair IDs, rather than raw participant IDs."""
        return list(self._pair_samples)

    def summary(self):
        filter_name = "Nilearn band-pass" if self.nilearn_filter else "none"
        norm_name = "per-session/per-region z-score" if self.normalise else "none"
        return (
            f"{self.name()} | progression subjects={len(self._progression_subjects)}, "
            f"pairs={len(self._pair_samples)}, parcellation={self.parcellation}, "
            f"timepoints={self.min_timepoints}\n"
            f"Preprocessing: filter={filter_name}, normalisation={norm_name}"
        )

    def _build_progression_pairs(self):
        pairs = []
        for subject_id in self.subjects:
            sessions = self.sessions.get(subject_id, [])
            diagnoses_by_session = self._diagnoses_by_session(subject_id, sessions)
            diagnoses = [diagnoses_by_session.get(session_id) for session_id in sessions]
            pair_indices = progression_pairs(diagnoses, self.allowed_progressions)
            if pair_indices:
                self._progression_subjects.append(subject_id)

            for initial_idx, progressed_idx in pair_indices:
                initial_session_id = sessions[initial_idx]
                progressed_session_id = sessions[progressed_idx]
                initial_timeseries = self.timeseries.get(subject_id, {}).get(initial_session_id)
                progressed_timeseries = self.timeseries.get(subject_id, {}).get(progressed_session_id)
                if initial_timeseries is None or progressed_timeseries is None:
                    continue
                pairs.append(
                    (
                        subject_id,
                        initial_session_id,
                        progressed_session_id,
                        diagnoses[initial_idx],
                        diagnoses[progressed_idx],
                        np.asarray(initial_timeseries),
                        np.asarray(progressed_timeseries),
                    )
                )

        if not pairs:
            raise ValueError("No ADNI-Long progression pairs with both timeseries were found.")

        all_sessions = [timeseries for pair in pairs for timeseries in pair[-2:]]
        self.min_timepoints = min(timeseries.shape[0] for timeseries in all_sessions)
        processed_sessions = {}
        for pair in pairs:
            subject_id, initial_id, progressed_id, _, _, initial_ts, progressed_ts = pair
            for session_id, timeseries in ((initial_id, initial_ts), (progressed_id, progressed_ts)):
                key = (subject_id, session_id)
                if key not in processed_sessions:
                    processed_sessions[key] = self._preprocess_session(subject_id, session_id, timeseries)

        for pair in pairs:
            subject_id, initial_id, progressed_id, initial_dx, progressed_dx, _, _ = pair
            sample_id = f"{subject_id}/{initial_id}/{progressed_id}"
            self._pair_classification[sample_id] = progressed_dx
            self._pair_samples[sample_id] = {
                "timeseries": np.stack(
                    (processed_sessions[(subject_id, initial_id)],
                     processed_sessions[(subject_id, progressed_id)]),
                    axis=0,
                ),
                "ABeta": self.abeta.get(subject_id, {}).get(progressed_id),
                "Tau": self.tau.get(subject_id, {}).get(progressed_id),
                "GMvol": self.gmvol.get(subject_id, {}).get(progressed_id),
                "GMvolAPC": self.gmvol_apc.get(subject_id),
                "subject_id": subject_id,
                "initial_session_id": initial_id,
                "progressed_session_id": progressed_id,
                "initial_diagnosis": initial_dx,
                "progressed_diagnosis": progressed_dx,
            }
        self.groups = [group for group in ("HC", "MCI", "AD") if self.get_groupSubjects(group)]

    def _preprocess_session(self, subject_id, session_id, timeseries):
        if timeseries.ndim != 2:
            raise ValueError(
                f"Timeseries for {subject_id}/{session_id} must be 2D; got {timeseries.shape}."
            )
        processed = np.asarray(timeseries[:self.min_timepoints], dtype=np.float32)
        if self.nilearn_filter is not None:
            processed = self.nilearn_filter.filter(processed.T).T
        if self.normalise:
            processed = self._normalise_session(processed)
        return processed

    def _diagnoses_by_session(self, subject_id, sessions):
        metadata = self.session_metadata.get(subject_id)
        if metadata is None or not {"session_id", "diagnosis"}.issubset(metadata.columns):
            return {}
        rows = metadata[["session_id", "diagnosis"]].dropna(subset=["session_id"])
        rows = rows.drop_duplicates(subset=["session_id"], keep="first")
        return {
            str(session_id): _normalise_diagnosis(diagnosis)
            for session_id, diagnosis in rows.itertuples(index=False, name=None)
            if str(session_id) in sessions
        }

    @staticmethod
    def _normalise_session(timeseries):
        means = timeseries.mean(axis=0, keepdims=True)
        scales = timeseries.std(axis=0, keepdims=True)
        scales = np.where(scales > 1e-12, scales, 1.0)
        return ((timeseries - means) / scales).astype(np.float32, copy=False)


def load_adni_long(data_dir=None, **kwargs):
    """Load preprocessed ADNI-Long progression pairs."""
    return ADNILongLoader(path=data_dir, **kwargs)


__all__ = [
    "ADNILongLoader",
    "DEFAULT_ALLOWED_PROGRESSIONS",
    "has_allowed_progression",
    "load_adni_long",
    "progression_pairs",
]
