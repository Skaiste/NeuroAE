# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, SpatialAttentionBlock, Upsample
from monai.utils import ensure_tuple_rep, optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

__all__ = ["AutoencoderKL"]


class AsymmetricPad(nn.Module):
    """
    Pad the input tensor asymmetrically along every spatial dimension.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
    """

    def __init__(self, spatial_dims: int) -> None:
        super().__init__()
        self.pad = (0, 1) * spatial_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, self.pad, mode="constant", value=0.0)
        return x


class AEKLDownsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
    """

    def __init__(self, spatial_dims: int, in_channels: int) -> None:
        super().__init__()
        self.pad = AsymmetricPad(spatial_dims=spatial_dims)

        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        return x


class AEKLResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, norm_num_groups: int, norm_eps: float, out_channels: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.nin_shortcut: nn.Module
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: number of input channels.
        channels: sequence of block output channels.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        blocks: List[nn.Module] = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Residual and downsampling blocks
        output_channel = channels[0]
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels) - 1

            for _ in range(self.num_res_blocks[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(AEKLDownsample(spatial_dims=spatial_dims, in_channels=input_channel))
        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=channels[-1],
                )
            )

            blocks.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=channels[-1],
                )
            )
        # Normalise and convert to latent size
        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=channels[-1], eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        use_convtranspose: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        reversed_block_out_channels = list(reversed(channels))

        blocks: List[nn.Module] = []

        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )
            blocks.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                if use_convtranspose:
                    blocks.append(
                        Upsample(
                            spatial_dims=spatial_dims, mode="deconv", in_channels=block_in_ch, out_channels=block_in_ch
                        )
                    )
                else:
                    post_conv = Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        out_channels=block_in_ch,
                        strides=1,
                        kernel_size=3,
                        padding=1,
                        conv_only=True,
                    )
                    blocks.append(
                        Upsample(
                            spatial_dims=spatial_dims,
                            mode="nontrainable",
                            in_channels=block_in_ch,
                            out_channels=block_in_ch,
                            interp_mode="nearest",
                            scale_factor=2.0,
                            post_conv=post_conv,
                            align_corners=None,
                        )
                    )

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class AutoencoderKLv2(nn.Module):
    """
    Autoencoder model with KL-regularized latent space based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        channels: number of output channels for each block.
        attention_levels: sequence of levels to add attention.
        latent_channels: latent embedding dimension.
        norm_num_groups: number of groups for the GroupNorm layers, channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        with_encoder_nonlocal_attn: if True use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: if True use non-local attention block in the decoder.
        time_shared: if True (only with ``spatial_dims=1``), input is interpreted as ``(B, R, T)`` and reshaped
            to ``(B*T, 1, R)`` so the model compresses regions ``R`` while preserving the time axis ``T``.
        use_checkpoint: if True, use activation checkpoint to save memory.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
        include_fc: whether to include the final linear layer in the attention block. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection in the attention block, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        time_shared: bool = False,
        use_checkpoint: bool = False,
        use_convtranspose: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in channels):
            raise ValueError("AutoencoderKL expects all channels being multiple of norm_num_groups")

        if len(channels) != len(attention_levels):
            raise ValueError("AutoencoderKL expects channels being same size of attention_levels")

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(channels))

        if len(num_res_blocks) != len(channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`channels`."
            )
        if time_shared:
            if spatial_dims != 1:
                raise ValueError("time_shared=True is only supported when spatial_dims=1.")
            if in_channels != 1 or out_channels != 1:
                raise ValueError(
                    "time_shared=True expects in_channels=1 and out_channels=1. "
                    "Provide input as (B, R, T)."
                )

        self.encoder: nn.Module = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.decoder: nn.Module = Decoder(
            spatial_dims=spatial_dims,
            channels=channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_convtranspose=use_convtranspose,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.quant_conv_mu = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.quant_conv_log_sigma = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.post_quant_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.latent_channels = latent_channels
        self.time_shared = time_shared
        self.use_checkpoint = use_checkpoint
        self.swfcd = None

    def set_loss_fn_params(self, params):
        self.loss_fn_params = params

    def set_swfcd(self, swfcd):
        self.swfcd = swfcd

    @staticmethod
    def _fold_time_into_batch(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if x.ndim != 3:
            raise ValueError(f"time_shared expects input shape (B, R, T), got {tuple(x.shape)}")
        bsz, regions, timepoints = x.shape
        x_folded = x.permute(0, 2, 1).reshape(bsz * timepoints, 1, regions).contiguous()
        return x_folded, (bsz, regions, timepoints)

    @staticmethod
    def _unfold_latent_from_batch(z: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        bsz, _, timepoints = shape
        return z.reshape(bsz, timepoints, z.shape[1], z.shape[2]).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _fold_latent_time_into_batch(z: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if z.ndim != 4:
            raise ValueError(f"time_shared expects latent shape (B, C, L, T), got {tuple(z.shape)}")
        bsz, channels, latent_len, timepoints = z.shape
        z_folded = z.permute(0, 3, 1, 2).reshape(bsz * timepoints, channels, latent_len).contiguous()
        return z_folded, (bsz, timepoints)

    @staticmethod
    def _unfold_recon_from_batch(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        bsz, timepoints = shape
        x_unfolded = x.reshape(bsz, timepoints, x.shape[1], x.shape[2]).permute(0, 2, 3, 1).contiguous()
        if x_unfolded.shape[1] == 1:
            return x_unfolded[:, 0]
        return x_unfolded

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        reshape_meta: tuple[int, int, int] | None = None
        x_in = x
        if self.time_shared:
            x_in, reshape_meta = self._fold_time_into_batch(x)

        if self.use_checkpoint:
            h = torch.utils.checkpoint.checkpoint(self.encoder, x_in, use_reentrant=False)
        else:
            h = self.encoder(x_in)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        if self.time_shared and reshape_meta is not None:
            z_mu = self._unfold_latent_from_batch(z_mu, reshape_meta)
            z_sigma = self._unfold_latent_from_batch(z_sigma, reshape_meta)

        return z_mu, z_sigma

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """
        From the mean and sigma representations resulting of encoding an image through the latent space,
        obtains a noise sample resulting from sampling gaussian noise, multiplying by the variance (sigma) and
        adding the mean.

        Args:
            z_mu: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] mean vector obtained by the encoder when you encode an image
            z_sigma: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] variance vector obtained by the encoder when you encode an image

        Returns:
            sample of shape Bx[Z_CHANNELS]x[LATENT SPACE SIZE]
        """
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes and decodes an input image.

        Args:
            x: BxCx[SPATIAL DIMENSIONS] tensor.

        Returns:
            reconstructed image, of the same shape as input
        """
        z_mu, _ = self.encode(x)
        reconstruction = self.decode(z_mu)
        return reconstruction

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Based on a latent space sample, forwards it through the Decoder.

        Args:
            z: Bx[Z_CHANNELS]x[LATENT SPACE SHAPE]

        Returns:
            decoded image tensor
        """
        reshape_meta: tuple[int, int] | None = None
        z_in = z
        if self.time_shared and z.ndim == 4:
            z_in, reshape_meta = self._fold_latent_time_into_batch(z)

        z_in = self.post_quant_conv(z_in)
        dec: torch.Tensor
        if self.use_checkpoint:
            dec = torch.utils.checkpoint.checkpoint(self.decoder, z_in, use_reentrant=False)
        else:
            dec = self.decoder(z_in)

        if self.time_shared and reshape_meta is not None:
            dec = self._unfold_recon_from_batch(dec, reshape_meta)
        return dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z)
        return reconstruction, z_mu, z_sigma, z

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        image = self.decode(z)
        return image

    def load_old_state_dict(self, old_state_dict: dict, verbose=False) -> None:
        """
        Load a state dict from an AutoencoderKL trained with [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels).

        Args:
            old_state_dict: state dict from the old AutoencoderKL model.
        """

        new_state_dict = self.state_dict()
        # if all keys match, just load the state dict
        if all(k in new_state_dict for k in old_state_dict):
            print("All keys match, loading state dict.")
            self.load_state_dict(old_state_dict)
            return

        if verbose:
            # print all new_state_dict keys that are not in old_state_dict
            for k in new_state_dict:
                if k not in old_state_dict:
                    print(f"key {k} not found in old state dict")
            # and vice versa
            print("----------------------------------------------")
            for k in old_state_dict:
                if k not in new_state_dict:
                    print(f"key {k} not found in new state dict")

        # copy over all matching keys
        for k in new_state_dict:
            if k in old_state_dict:
                new_state_dict[k] = old_state_dict.pop(k)

        # fix the attention blocks
        attention_blocks = [k.replace(".attn.to_q.weight", "") for k in new_state_dict if "attn.to_q.weight" in k]
        for block in attention_blocks:
            new_state_dict[f"{block}.attn.to_q.weight"] = old_state_dict.pop(f"{block}.to_q.weight")
            new_state_dict[f"{block}.attn.to_k.weight"] = old_state_dict.pop(f"{block}.to_k.weight")
            new_state_dict[f"{block}.attn.to_v.weight"] = old_state_dict.pop(f"{block}.to_v.weight")
            new_state_dict[f"{block}.attn.to_q.bias"] = old_state_dict.pop(f"{block}.to_q.bias")
            new_state_dict[f"{block}.attn.to_k.bias"] = old_state_dict.pop(f"{block}.to_k.bias")
            new_state_dict[f"{block}.attn.to_v.bias"] = old_state_dict.pop(f"{block}.to_v.bias")

            # old version did not have a projection so set these to the identity
            new_state_dict[f"{block}.attn.out_proj.weight"] = torch.eye(
                new_state_dict[f"{block}.attn.out_proj.weight"].shape[0]
            )
            new_state_dict[f"{block}.attn.out_proj.bias"] = torch.zeros(
                new_state_dict[f"{block}.attn.out_proj.bias"].shape
            )

        # fix the upsample conv blocks which were renamed postconv
        for k in new_state_dict:
            if "postconv" in k:
                old_name = k.replace("postconv", "conv")
                new_state_dict[k] = old_state_dict.pop(old_name)
        if verbose:
            # print all remaining keys in old_state_dict
            print("remaining keys in old_state_dict:", old_state_dict.keys())
        self.load_state_dict(new_state_dict, strict=True)
    
    def loss(self, x, model_output):
        x_hat, z_mu, z_sigma, _ = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        beta = float(self.loss_fn_params.get("beta", 1.0))
        # if selected error per feature, we are averaging everything
        if error_per_feature:
            # recon: mean mse loss
            recon = F.mse_loss(x_hat, x, reduction="mean")

        # if selected error per sample, we are summing everything
        else:
            # recon: sum over features per sample, then mean over batch
            recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
            recon = recon.sum(dim=1).mean()

        kld = -0.5 * (1 + z_sigma - z_mu.pow(2) - z_sigma.exp())
        kld = kld.flatten(1).sum(dim=1).mean() / z_sigma.size(1)

        loss = {
            'loss': recon + beta * kld,
            'recon': recon, 
            'kld': kld
        }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, x_hat)
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss['swfcd_rmse'] = swfcd['rmse']
            loss['loss'] += swfcd_beta * swfcd['rmse']

        return loss


"""
    Prediction head models
"""
class PredHeadAvg(nn.Module):
    """ Most simple model
         - averaging across time point dimension 
         - applying linear layer to predict levels
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.head = nn.Linear(latent_dim, output_dim, bias=True)

    def forward(self, z):
        x = z.mean(dim=-1)
        return self.head(x)
    
class PredHeadConv(nn.Module):
    """ Convolutional model
        - temporal conv model with average pooling
        - linear layer for level prediction
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # since selected latent dimension is usually a small number
        # we can double it, another suggestion would be to get the middle number
        # in between latent and output dimensions, but that would bloat the model
        self.hidden_dim = latent_dim * 2
        self.temporal = nn.Sequential(
            nn.Conv1d(latent_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_dim, output_dim)
        )
    def forward(self, z):
        x = self.temporal(z)
        return self.head(x)
    
class TemporalAttentionPooling(nn.Module):
    """ Model focuses on time points that are more important """
    def __init__(self, latent_dim):
        super().__init__()
        self.attn = nn.Linear(latent_dim, 1)

    def forward(self, z): # z: (B, L, T)
        z_t = z.transpose(1, 2)      # (B, T, L)
        scores = self.attn(z_t)      # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * z_t).sum(dim=1)  # (B, L)
        return pooled


class PredHeadTemporalPool(nn.Module):
    """ Temporal pooling model
         - use temporal pooling to reduce time point dimension to 1
         - applying linear layer to predict levels
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.pool = TemporalAttentionPooling(latent_dim)
        self.head = nn.Linear(latent_dim, output_dim, bias=True)

    def forward(self, z):
        x = self.pool(z)
        return self.head(x)
    
class GatedTemporalAttentionPooling(nn.Module):
    """ Richer version of attention pooling, uses two projections:
         - one to model the content
         - one to act like a gate (suppress or amplify parts of the timepoint representation)
        It can capture:
         - this timepoint is important only if certain latent features are active
         - interactions between latent content and importance
         - more selective weighting over time
        it often gives sharper and more meaningful attention weights.
    """
    def __init__(self, latent_dim: int, attn_dim: int):
        super().__init__()
        print(f"GatedTemporalAttentionPooling {latent_dim=} {attn_dim=}")
        self.V = nn.Linear(latent_dim, attn_dim)
        self.U = nn.Linear(latent_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, z):   # z: (B, L, T)
        breakpoint()
        z_t = z.transpose(1, 2)  # (B, T, L)
        v = torch.tanh(self.V(z_t))        # (B, T, A)
        u = torch.sigmoid(self.U(z_t))     # (B, T, A)
        h = v * u                          # (B, T, A)
        scores = self.w(h)                 # (B, T, 1)
        weights = torch.softmax(scores, dim=1)   # (B, T, 1)
        pooled = (weights * z_t).sum(dim=1)      # (B, L)
        return pooled, weights.squeeze(-1)
    
class PredHeadGatedTemporalPool(nn.Module):
    """ Gated Temporal pooling model
         - use gated temporal pooling to reduce time point dimension to 1
         - applying linear layer to predict levels
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # like the convolutional hidden dim, same applies
        self.attention_dim = latent_dim * 2
        self.pool = GatedTemporalAttentionPooling(latent_dim, self.attention_dim)
        self.head = nn.Linear(latent_dim, output_dim, bias=True)

    def forward(self, z):
        x, weights = self.pool(z)
        # maybe return the weights? for now will refrain
        return self.head(x)

class AutoencoderKLv2ABT(AutoencoderKLv2):
    """ AutoencoderKLv2 + linear temporal models for ABeta and Tau level prediction """
    def __init__(self, spatial_dims, pred_out, pred_head_type="gated_temp_pool", in_channels = 1, out_channels = 1, num_res_blocks = (2, 2, 2, 2), channels = (32, 64, 64, 64), attention_levels = (False, False, True, True), latent_channels = 3, norm_num_groups = 32, norm_eps = 0.000001, with_encoder_nonlocal_attn = True, with_decoder_nonlocal_attn = True, time_shared = False, use_checkpoint = False, use_convtranspose = False, include_fc = True, use_combined_linear = False, use_flash_attention = False):
        super().__init__(spatial_dims, in_channels, out_channels, num_res_blocks, channels, attention_levels, latent_channels, norm_num_groups, norm_eps, with_encoder_nonlocal_attn, with_decoder_nonlocal_attn, time_shared, use_checkpoint, use_convtranspose, include_fc, use_combined_linear, use_flash_attention)

        pred_head_idx = {
            "avg": PredHeadAvg,
            "conv": PredHeadConv,
            "temp_pool": PredHeadTemporalPool,
            "gated_temp_pool": PredHeadGatedTemporalPool
        }
        if pred_head_type not in pred_head_idx:
            raise ValueError(f"Selected prediction head type - '{pred_head_type}' is not available.")
        
        self.abeta_pred = pred_head_idx[pred_head_type](latent_channels, pred_out)
        self.tau_pred = pred_head_idx[pred_head_type](latent_channels, pred_out)

    def forward(self, x):
        recon, z_mu, z_sigma, z = super().forward(x)
        breakpoint()
        abeta = self.abeta_pred(z)
        tau = self.tau_pred(z)
        # insert these values before the z
        return recon, z_mu, z_sigma, abeta, tau, z
    
    
    def loss(self, x, x_abeta, x_tau, model_output):
        x_hat, z_mu, z_sigma, z_abeta, z_tau, _ = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        beta = float(self.loss_fn_params.get("beta", 1.0))
        # if selected error per feature, we are averaging everything
        if error_per_feature:
            # recon: mean mse loss
            recon = F.mse_loss(x_hat, x, reduction="mean")

        # if selected error per sample, we are summing everything
        else:
            # recon: sum over features per sample, then mean over batch
            recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
            recon = recon.sum(dim=1).mean()

        kld = -0.5 * (1 + z_sigma - z_mu.pow(2) - z_sigma.exp())
        kld = kld.flatten(1).sum(dim=1).mean() / z_sigma.size(1)

        # calculate loss for ABeta and Tau prediction
        abeta_loss = F.smooth_l1_loss(z_abeta, x_abeta, reduction="mean", beta=1.0)
        tau_loss = F.smooth_l1_loss(z_tau, x_tau, reduction="mean", beta=1.0)

        loss = {
            'loss': recon + beta * kld + abeta_loss + tau_loss,
            'recon': recon,
            'kld': kld,
            'abeta_loss': abeta_loss,
            'tau_loss': tau_loss
        }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, x_hat)
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss['swfcd_rmse'] = swfcd['rmse']
            loss['loss'] += swfcd_beta * swfcd['rmse']

        return loss