import torch

from ..load_data import ADNIDataset


class SwFCD:
    def __init__(self, dataset: ADNIDataset, window_size: int, window_step: int, eps: float = 1e-8):
        self.dataset = dataset
        self.window_size = int(window_size)
        self.window_step = int(window_step)
        self.eps = float(eps)

        if self.window_size <= 1:
            raise ValueError("window_size must be > 1 for correlation-based SWFCD")
        if self.window_step <= 0:
            raise ValueError("window_step must be > 0")

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

        iu = torch.triu_indices(n_roi, n_roi, offset=1, device=x_btr.device)
        fc_vec = fc[..., iu[0], iu[1]]

        # FCD: corr between FC vectors across windows.
        fc_vec = self._safe_standardize_last_dim(fc_vec)
        fcd = torch.matmul(fc_vec, fc_vec.transpose(-1, -2))

        iu_w = torch.triu_indices(n_windows, n_windows, offset=1, device=x_btr.device)
        return fcd[..., iu_w[0], iu_w[1]]

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

    def apply(self, x: torch.Tensor, x_hat: torch.Tensor):
        x_btr = self.ensure_correct_dim(x)
        x_hat_btr = self.ensure_correct_dim(x_hat)

        if x_btr is None or x_hat_btr is None:
            # raise ValueError("SWFCD cannot be computed when dataset.fc_input is True")
            return None

        x_vec = self._swfcd_vector_from_bold(x_btr)
        x_hat_vec = self._swfcd_vector_from_bold(x_hat_btr)

        pearson, mad, rmse = self._pairwise_metrics(x_vec, x_hat_vec)

        return {
            "pearson": pearson,
            "mad": mad,
            "rmse": rmse,
        }
