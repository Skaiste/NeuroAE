import os
from typing import Optional

import torch

from ..data import BaseTimeseriesDataset


class SwFCD:
    def __init__(self, dataset: BaseTimeseriesDataset, window_size: int, window_step: int, eps: float = 1e-8):
        self.dataset = dataset
        self.window_size = int(window_size)
        self.window_step = int(window_step)
        self.eps = float(eps)

        if self.window_size <= 1:
            raise ValueError("window_size must be > 1 for correlation-based SWFCD")
        if self.window_step <= 0:
            raise ValueError("window_step must be > 0")
        self._triu_cache = {}
        self.debug = os.environ.get("NEUROAE_SWFCD_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    def ensure_correct_dim(self, x: torch.Tensor):# -> torch.Tensor | None:
        """Convert supported dataset layouts to (B, T, R)."""
        if self.dataset.fc_input:
            return None

        if not torch.is_tensor(x):
            raise TypeError("SwFCD expects torch.Tensor inputs")

        if self.dataset.timepoints_as_samples:
            # current shape: (B*T, R) -> (B, T, R)
            return x.reshape(-1, self.dataset.original_shape[0], self.dataset.original_shape[1])

        if self.dataset.flatten:
            # current shape: (B, R*T) or (B, T*R) -> (B, ?, ?)
            x = x.reshape(-1, self.dataset.original_shape[0], self.dataset.original_shape[1])

        if not self.dataset.transpose:
            # current shape: (B, R, T) -> (B, T, R)
            x = x.transpose(1, 2)

        return x

    def _safe_standardize_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        denom = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        return x / denom

    def _triu_indices(self, rows: int, cols: int, offset: int, device: torch.device) -> torch.Tensor:
        key = (int(rows), int(cols), int(offset), str(device))
        cached = self._triu_cache.get(key)
        if cached is None:
            cached = torch.triu_indices(rows, cols, offset=offset, device=device)
            self._triu_cache[key] = cached
        return cached

    def _describe_input(self, x_btr: torch.Tensor) -> str:
        return (
            f"shape={tuple(x_btr.shape)}, dtype={x_btr.dtype}, device={x_btr.device}, "
            f"window_size={self.window_size}, window_step={self.window_step}, "
            f"dataset(flatten={getattr(self.dataset, 'flatten', None)}, "
            f"transpose={getattr(self.dataset, 'transpose', None)}, "
            f"timepoints_as_samples={getattr(self.dataset, 'timepoints_as_samples', None)}, "
            f"fc_input={getattr(self.dataset, 'fc_input', None)}, "
            f"original_shape={getattr(self.dataset, 'original_shape', None)})"
        )

    def _validate_bold_input(self, x_btr: torch.Tensor) -> None:
        if x_btr.ndim != 3:
            raise ValueError(f"SwFCD expects (B, T, R) after reshaping, got {tuple(x_btr.shape)}")

        batches, t_len, n_rois = x_btr.shape
        if batches <= 0:
            raise ValueError(f"SwFCD received an empty batch: {tuple(x_btr.shape)}")
        if t_len < self.window_size:
            raise ValueError(
                f"SwFCD requires at least {self.window_size} timepoints, got {t_len}. "
                f"Input context: {self._describe_input(x_btr)}"
            )
        if n_rois < 2:
            raise ValueError(
                f"SwFCD requires at least 2 ROIs, got {n_rois}. "
                f"Input context: {self._describe_input(x_btr)}"
            )
        if not torch.isfinite(x_btr).all():
            nan_count = int(torch.isnan(x_btr).sum().item())
            inf_count = int(torch.isinf(x_btr).sum().item())
            raise ValueError(
                f"SwFCD input contains non-finite values (nan={nan_count}, inf={inf_count}). "
                f"Input context: {self._describe_input(x_btr)}"
            )

    def _window_count(self, t_len: int) -> int:
        return 1 + (t_len - self.window_size) // self.window_step

    def _estimate_impl2_bytes(self, x_btr: torch.Tensor) -> tuple[int, int]:
        batches, t_len, n_rois = x_btr.shape
        n_windows = self._window_count(t_len)
        bytes_per_elem = max(x_btr.element_size(), 4)
        corr_mtx = batches * n_windows * n_rois * n_rois * bytes_per_elem
        cotsampling = batches * n_windows * max(n_windows - 1, 0) // 2 * bytes_per_elem
        return n_windows, corr_mtx + cotsampling

    def _raise_if_impl2_too_large(self, x_btr: torch.Tensor) -> None:
        n_windows, estimated_bytes = self._estimate_impl2_bytes(x_btr)
        limit_bytes = int(float(os.environ.get("NEUROAE_SWFCD_MAX_MB", "1024")) * 1024 * 1024)
        if estimated_bytes > limit_bytes:
            estimated_mb = estimated_bytes / (1024 * 1024)
            limit_mb = limit_bytes / (1024 * 1024)
            raise MemoryError(
                "SwFCD would allocate too much memory before computing correlations. "
                f"Estimated working set for `_swfcd_vector_from_bold_2`: {estimated_mb:.1f} MB "
                f"(limit {limit_mb:.1f} MB, n_windows={n_windows}). "
                f"Input context: {self._describe_input(x_btr)}. "
                "Reduce batch size / ROI count, increase window_step, or use the more memory-efficient implementation."
            )
        if self.debug:
            estimated_mb = estimated_bytes / (1024 * 1024)
            print(
                "SWFCD DEBUG:",
                f"n_windows={n_windows}",
                f"estimated_impl2_memory={estimated_mb:.1f}MB",
                self._describe_input(x_btr),
            )

    def _swfcd_vector_from_bold(self, x_btr: torch.Tensor) -> torch.Tensor:
        """Compute SWFCD vector per batch element from BOLD time series (B, T, R)."""
        bsz, t_len, n_roi = x_btr.shape
        if t_len < self.window_size:
            return x_btr.new_zeros((bsz, 0))

        windows = x_btr.unfold(dimension=1, size=self.window_size, step=self.window_step)
        # unfold gives (B, W, R, window_size). Reorder to (B, W, window_size, R)
        windows = windows.permute(0, 1, 3, 2)
        n_windows = windows.shape[1]

        if n_windows < 2:
            return x_btr.new_zeros((bsz, 0))

        # FC per window: corr over time for each ROI pair.
        # windows_t: (B, W, R, window_size)
        windows_t = windows.permute(0, 1, 3, 2)
        windows_t = self._safe_standardize_last_dim(windows_t)
        fc = torch.matmul(windows_t, windows_t.transpose(-1, -2))

        iu = self._triu_indices(n_roi, n_roi, offset=1, device=x_btr.device)
        fc_vec = fc[..., iu[0], iu[1]]

        # FCD: corr between FC vectors across windows.
        fc_vec = self._safe_standardize_last_dim(fc_vec)
        fcd = torch.matmul(fc_vec, fc_vec.transpose(-1, -2))

        iu_w = self._triu_indices(n_windows, n_windows, offset=1, device=x_btr.device)
        return fcd[..., iu_w[0], iu_w[1]]


    def _swfcd_vector_from_bold_2(self, x_btr: torch.Tensor) -> torch.Tensor:
        x_btr = x_btr.transpose(2,1)
        batches, t_max, n_rois = x_btr.shape

        self._validate_bold_input(x_btr)
        # self._raise_if_impl2_too_large(x_btr)

        last_window_start = t_max - self.window_size
        n_windows = self._window_count(t_max)

        corr_mtrxs = x_btr.new_zeros((batches, n_windows, n_rois, n_rois))

        for b in range(batches):
            w_idx = 0
            for t in range(0, last_window_start + 1, self.window_step):
                w_data = x_btr[b, t:t + self.window_size, :].T
                cm = torch.corrcoef(w_data)
                cm = torch.nan_to_num(cm, nan=0.0)
                cm.fill_diagonal_(1.0)
                corr_mtrxs[b, w_idx] = cm
                w_idx += 1

        rows, cols = torch.tril_indices(n_rois, n_rois, offset=-1)
        lower_triangular_parts = corr_mtrxs[:, :, rows, cols]
        
        cotsampling = x_btr.new_zeros((batches, n_windows * (n_windows - 1) // 2))
        rows, cols = torch.triu_indices(n_windows, n_windows, offset=1)
        for b in range(batches):
            corr = torch.corrcoef(lower_triangular_parts[b])
            cotsampling[b] = corr[rows, cols]

        return cotsampling


    def _pairwise_metrics(self, x_vec: torch.Tensor, x_hat_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        common_len = min(x_vec.shape[-1], x_hat_vec.shape[-1])
        if common_len == 0:
            nan = x_vec.new_tensor(float("nan"))
            return nan, nan, nan

        x_vec = x_vec[..., :common_len]
        x_hat_vec = x_hat_vec[..., :common_len]

        x_centered = x_vec - x_vec.mean(dim=-1, keepdim=True)
        x_hat_centered = x_hat_vec - x_hat_vec.mean(dim=-1, keepdim=True)

        num = (x_centered * x_hat_centered).sum(dim=-1)
        den = (
            torch.linalg.norm(x_centered, dim=-1).clamp_min(self.eps)
            * torch.linalg.norm(x_hat_centered, dim=-1).clamp_min(self.eps)
        )
        pearson = num / den

        mad = (x_vec - x_hat_vec).abs().mean(dim=-1)
        rmse = torch.sqrt(((x_vec - x_hat_vec) ** 2).mean(dim=-1))

        # Keep compatibility with loss code expecting a scalar tensor.
        return pearson.mean(), mad.mean(), rmse.mean()

    def vectorize(self, x: torch.Tensor, *, track_grad: Optional[bool] = None) -> Optional[torch.Tensor]:
        x_btr = self.ensure_correct_dim(x)
        if x_btr is None:
            return None

        if track_grad is None:
            track_grad = bool(x.requires_grad)

        if track_grad:
            try:
                return self._swfcd_vector_from_bold_2(x_btr)
            except Exception as exc:
                raise RuntimeError(
                    f"SwFCD vectorization failed with gradients enabled. {self._describe_input(x_btr)}"
                ) from exc

        with torch.no_grad():
            try:
                return self._swfcd_vector_from_bold_2(x_btr)
            except Exception as exc:
                raise RuntimeError(
                    f"SwFCD vectorization failed under no_grad. {self._describe_input(x_btr)}"
                ) from exc

    def apply(self, x: Optional[torch.Tensor], x_hat: torch.Tensor, x_vec: Optional[torch.Tensor] = None):
        if x_vec is None:
            if x is None:
                raise ValueError("SwFCD.apply requires either x or x_vec.")
            x_vec = self.vectorize(x, track_grad=False)
        x_hat_btr = self.ensure_correct_dim(x_hat)

        if x_vec is None or x_hat_btr is None:
            # raise ValueError("SWFCD cannot be computed when dataset.fc_input is True")
            return None

        try:
            x_hat_vec = self._swfcd_vector_from_bold_2(x_hat_btr)
        except Exception as exc:
            raise RuntimeError(
                f"SwFCD apply failed while vectorizing reconstruction. {self._describe_input(x_hat_btr)}"
            ) from exc

        pearson, mad, rmse = self._pairwise_metrics(x_vec, x_hat_vec)

        return {
            "pearson": pearson,
            "mad": mad,
            "rmse": rmse,
        }
