"""HyDen MoGe-2 (DINOv2 ViT-L/14 + MobileNetV2 CNN_Large) metric 3D point and depth model.

    from vendor.meta_moge2.hyden_moge2 import (
        DEFAULT_CHECKPOINT, CHECKPOINT_NUM_KEYS, HyDenMoGe, load_hyden_moge2,
        validate_raw_rgb, recover_focal_shift, recover_intrinsics, fov_deg_from_intrinsics,
    )

Adapted from `facebookresearch/metadepth` at commit `810b77c3e56712813a0de42a130cfbe6f5b19b90`
(`mogev2/moge_v2.py`, `mogev2/modules.py`, `mogev2/utils.py`, `cnn/encoder.py`, and
`example_code/hyden_load_and_inference.py`); see `vendor/meta_moge2/README.md` and
`vendor/meta_moge2/LICENSE`. Wired into no ComfyUI node, keep-out, reference, or load step.

**Input contract (raw-RGB guard).** Unlike DA2 (which expects ImageNet-normalized inputs),
HyDen MoGe-2 applies ImageNet `(x - mean) / std` inside its DINOv2 encoder branch after
resizing to `(518, 518)`, while its `CNN_Large` branch runs on the full-resolution raw `[0, 1]`
RGB image and fuses with the 4 ViT intermediate layers (`[5, 11, 17, 23]`). Passing `uint8`
`[0, 255]` tensors or pre-normalized ImageNet `[-2.1, 2.6]` tensors is refused by
`validate_raw_rgb`.
"""
from __future__ import annotations

import functools
import itertools
import math
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import Conv2dNormActivation

#: Default local checkpoint path, resolved relative to the current user's home directory.
DEFAULT_CHECKPOINT: Path = (
    Path.home()
    / "Storage"
    / "facebook_hyden-mogev2-metric-point"
    / "hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth"
)

#: Exact number of state_dict keys (738 parameters + 215 buffers) in the metric-point checkpoint.
#: Measured on `hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth` (483 keys without the
#: HyDen `CNN_Large` fusion branch; 953 keys with `CNN_Large`, `image_feats`, and `scale_feats`).
CHECKPOINT_NUM_KEYS: int = 953

#: Sum of output channels across the 5 `CNN_Large` stages: 32 + 32 + 64 + 160 + 288 = 576.
CNN_TOTAL_CHANNELS: int = 576
#: Output channels of the final `CNN_Large` stage (`stage4`).
CNN_LAST_CHANNEL: int = 288

#: Tolerance on `[0, 1]` input bounds for floating-point rounding before refusal.
RAW_RGB_EPS: float = 1e-4


def validate_raw_rgb(image: torch.Tensor) -> None:
    """Refuse anything other than raw unnormalized `[0, 1]` `float32` RGB `(B, 3, H, W)`."""
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"raw-RGB guard: expected torch.Tensor, got {type(image).__name__}")
    if image.dtype != torch.float32:
        raise ValueError(
            f"raw-RGB guard: expected float32 tensor in [0, 1], got dtype {image.dtype} "
            "(do not pass uint8 [0, 255] tensors)"
        )
    if image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3 or image.shape[2] < 14 or image.shape[3] < 14:
        raise ValueError(
            f"raw-RGB guard: expected shape (B, 3, H>=14, W>=14), got {tuple(image.shape)}"
        )
    if not bool(torch.isfinite(image).all()):
        raise ValueError("raw-RGB guard: input image contains NaN or infinite values")
    lo = float(image.min().item())
    hi = float(image.max().item())
    if lo < -RAW_RGB_EPS or hi > 1.0 + RAW_RGB_EPS:
        raise ValueError(
            f"raw-RGB guard: expected raw RGB in [0, 1], got range [{lo:.4f}, {hi:.4f}] "
            "(do not pass pre-normalized ImageNet [-2.1, 2.6] or unscaled [0, 255] values)"
        )


# --------------------------------------------------------- geometry & recovery

def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Normalized view-plane UV grid of shape `(H, W, 2)`."""
    if aspect_ratio is None:
        aspect_ratio = width / height
    diag = (1.0 + aspect_ratio ** 2) ** 0.5
    span_x = aspect_ratio / diag
    span_y = 1.0 / diag
    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u_grid, v_grid], dim=-1)


def intrinsics_from_focal_center(
    fx: float | torch.Tensor,
    fy: float | torch.Tensor,
    cx: float | torch.Tensor = 0.5,
    cy: float | torch.Tensor = 0.5,
) -> torch.Tensor:
    """Build normalized `(B, 3, 3)` camera intrinsics matrix from `(fx, fy, cx, cy)`."""
    if not isinstance(fx, torch.Tensor):
        fx = torch.tensor([float(fx)], dtype=torch.float32)
    if not isinstance(fy, torch.Tensor):
        fy = torch.tensor([float(fy)], dtype=fx.dtype, device=fx.device)
    b = fx.shape[0]
    intr = torch.zeros(b, 3, 3, device=fx.device, dtype=fx.dtype)
    intr[:, 0, 0] = fx
    intr[:, 1, 1] = fy
    intr[:, 0, 2] = cx
    intr[:, 1, 2] = cy
    intr[:, 2, 2] = 1.0
    return intr


def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject `(B, H, W)` depth map with `(B, 3, 3)` normalized intrinsics to `(B, H, W, 3)`."""
    _, h, w = depth.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = x / w
    y = y / h
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]
    pts_x = (x[None] - cx) * depth / fx
    pts_y = (y[None] - cy) * depth / fy
    return torch.stack([pts_x, pts_y, depth], dim=-1)


def _solve_optimal_shift(uv: np.ndarray, points: np.ndarray, focal: float) -> float:
    uv = uv.reshape(-1, 2)
    points = points.reshape(-1, 3)
    uv_norm = np.linalg.norm(uv, axis=-1)
    valid = uv_norm > 1e-6
    if valid.sum() < 2:
        return 0.0
    target_z = focal * np.linalg.norm(points[valid, :2], axis=-1) / uv_norm[valid]
    return float(np.median(target_z - points[valid, 2]))


def _solve_optimal_focal_shift(uv: np.ndarray, points: np.ndarray) -> tuple[float, float]:
    uv = uv.reshape(-1, 2)
    points = points.reshape(-1, 3)
    xy = points[:, :2]
    z = points[:, 2]
    mat_a = np.stack([xy, -uv], axis=-1).reshape(-1, 2)
    vec_b = (uv * z[:, None]).reshape(-1)
    try:
        sol, _, _, _ = np.linalg.lstsq(mat_a, vec_b, rcond=None)
        focal = float(sol[0])
        shift = float(sol[1])
    except np.linalg.LinAlgError:
        focal, shift = 1.0, 0.0
    if focal <= 0.0:
        focal = 1.0
    return shift, focal


def recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor | None = None,
    focal: torch.Tensor | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover `(focal, shift)` tensors of shape `(...)` from point map `(..., H, W, 3)`."""
    shape = points.shape
    height, width = shape[-3], shape[-2]
    pts_flat = points.reshape(-1, height, width, 3)
    mask_flat = None if mask is None else mask.reshape(-1, height, width)
    focal_flat = None if focal is None else focal.reshape(-1)
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    pts_lr = F.interpolate(
        pts_flat.permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).permute(0, 2, 3, 1)
    uv_lr = (
        F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest")
        .squeeze(0)
        .permute(1, 2, 0)
    )
    mask_lr = (
        None
        if mask_flat is None
        else F.interpolate(
            mask_flat.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest"
        ).squeeze(1)
        > 0
    )

    uv_np = uv_lr.cpu().numpy()
    pts_np = pts_lr.detach().cpu().numpy()
    focal_np = None if focal_flat is None else focal_flat.detach().cpu().numpy()
    mask_np = None if mask_lr is None else mask_lr.cpu().numpy()

    shifts: list[float] = []
    focals: list[float] = []
    for i in range(pts_flat.shape[0]):
        pts_i = pts_np[i] if mask_np is None else pts_np[i][mask_np[i]]
        uv_i = uv_np if mask_np is None else uv_np[mask_np[i]]
        if uv_i.shape[0] < 2:
            focals.append(1.0)
            shifts.append(0.0)
            continue
        if focal_flat is None:
            s_i, f_i = _solve_optimal_focal_shift(uv_i, pts_i)
            focals.append(float(f_i))
        else:
            assert focal_np is not None
            s_i = _solve_optimal_shift(uv_i, pts_i, float(focal_np[i]))
        shifts.append(float(s_i))

    shift_t = torch.tensor(shifts, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    if focal_flat is None:
        focal_t = torch.tensor(focals, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        focal_t = focal_flat.reshape(shape[:-3])
    return focal_t, shift_t


def recover_intrinsics(
    points: torch.Tensor,
    mask_binary: torch.Tensor | None = None,
    fov_x: float | torch.Tensor | None = None,
    aspect_ratio: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Recover `(shifted_points, depth, intrinsics, mask_binary)` from predicted `(B, H, W, 3)` points."""
    if aspect_ratio is None:
        aspect_ratio = points.shape[-2] / points.shape[-3]
    if fov_x is None:
        focal, shift = recover_focal_shift(points, mask_binary)
    else:
        fov_t = torch.as_tensor(fov_x, device=points.device, dtype=points.dtype)
        tan_half = torch.tan(torch.deg2rad(fov_t / 2.0))
        ar_t = torch.as_tensor(aspect_ratio, device=points.device, dtype=points.dtype)
        focal = ar_t / torch.sqrt(1.0 + ar_t.pow(2)) / tan_half
        if focal.ndim == 0:
            focal = focal[None].expand(points.shape[0])
        _, shift = recover_focal_shift(points, mask_binary, focal=focal)

    ar_factor = torch.sqrt(
        torch.as_tensor(1.0 + aspect_ratio ** 2, device=points.device, dtype=points.dtype)
    )
    ar_t = torch.as_tensor(aspect_ratio, device=points.device, dtype=points.dtype)
    fx = focal / 2.0 * ar_factor / ar_t
    fy = focal / 2.0 * ar_factor
    intrinsics = intrinsics_from_focal_center(fx, fy, 0.5, 0.5)

    shifted = points.clone()
    shifted[..., 2] = shifted[..., 2] + shift[..., None, None]
    if mask_binary is not None:
        mask_binary = mask_binary & (shifted[..., 2] > 0)
    depth = shifted[..., 2].clone()
    return shifted, depth, intrinsics, mask_binary


def fov_deg_from_intrinsics(intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(fov_x_deg, fov_y_deg)` of shape `(B,)` in degrees from `(B, 3, 3)` normalized intrinsics."""
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    fov_x = torch.rad2deg(2.0 * torch.atan(0.5 / fx))
    fov_y = torch.rad2deg(2.0 * torch.atan(0.5 / fy))
    return fov_x, fov_y


# ----------------------------------------------------------- DINOv2 ViT-L/14

class _Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, c)))


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads)
        self.ls1 = _LayerScale(dim, init_values=init_values)
        self.drop_path1 = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = _LayerScale(dim, init_values=init_values)
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 518, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 1024) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        init_values: float = 1.0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = _PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList(
            [_Block(dim=embed_dim, num_heads=num_heads, init_values=init_values) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        prev_dtype = x.dtype
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        cls_pos = pos_embed[:, 0]
        patch_pos = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size + self.interpolate_offset
        h0 = h // self.patch_size + self.interpolate_offset
        sqrt_n = math.sqrt(n)
        patch_pos = F.interpolate(
            patch_pos.reshape(1, int(sqrt_n), int(sqrt_n), dim).permute(0, 3, 1, 2),
            scale_factor=(float(w0) / sqrt_n, float(h0) / sqrt_n),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((cls_pos.unsqueeze(0), patch_pos), dim=1).to(prev_dtype)

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Sequence[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        take = set(n)
        outputs: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take:
                outputs.append(self.norm(x))
        return [(out[:, 1:], out[:, 0]) for out in outputs]


class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        intermediate_layers: Sequence[int] = (5, 11, 17, 23),
        dim_out: int = 1024,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.intermediate_layers = list(intermediate_layers)
        self.patch_size = patch_size
        self.backbone = _DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.output_projections = nn.ModuleList(
            [nn.Conv2d(embed_dim, dim_out, kernel_size=1, stride=1, padding=0) for _ in self.intermediate_layers]
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        image: torch.Tensor,
        token_rows: int,
        token_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_14 = F.interpolate(
            image,
            (int(token_rows) * self.patch_size, int(token_cols) * self.patch_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        image_14 = (image_14 - self.image_mean) / self.image_std
        layers = self.backbone.get_intermediate_layers(image_14, n=self.intermediate_layers)
        projected = torch.stack(
            [
                proj(
                    feat.permute(0, 2, 1)
                    .unflatten(2, (int(token_rows), int(token_cols)))
                    .contiguous()
                )
                for proj, (feat, _) in zip(self.output_projections, layers)
            ],
            dim=1,
        ).sum(dim=1)
        return projected, layers[-1][1]


# --------------------------------------------------- MobileNetV2 CNN_Large

class _InvertedResidual(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        group_size: int = 1,
    ) -> None:
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.res_conn_processor = nn.Identity() if self.use_res_connect else None
        self.pw = (
            Conv2dNormActivation(
                in_channels,
                hidden_dim,
                kernel_size=1,
                stride=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU,
                bias=False,
            )
            if expand_ratio != 1
            else None
        )
        self.dw = Conv2dNormActivation(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            groups=hidden_dim // group_size,
            norm_layer=None,
            activation_layer=None,
            bias=False,
        )
        self.se = None
        self.pwl = Conv2dNormActivation(
            hidden_dim,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(x) if self.pw is not None else x
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y = torch.add(y, x)
            if self.res_conn_processor is not None:
                y = self.res_conn_processor(y)
        return y


class CNNLarge(nn.Module):
    """5-stage MobileNetV2-style CNN encoder (`32, 32, 64, 160, 288` channels)."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        ir = partial(_InvertedResidual, kernel_size=3)
        ir5 = partial(_InvertedResidual, kernel_size=5)
        ir_group = partial(_InvertedResidual, kernel_size=3, group_size=32)
        ir5_group = partial(_InvertedResidual, kernel_size=5, group_size=32)

        first_conv = Conv2dNormActivation(
            in_channels,
            32,
            stride=2,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            bias=True,
        )
        stages = [
            nn.Sequential(first_conv, ir_group(32, 32, 2), ir_group(32, 32, 2), ir_group(32, 32, 2)),
            nn.Sequential(
                ir5_group(32, 32, 4, stride=2),
                ir_group(32, 32, 2),
                ir_group(32, 32, 2),
                ir_group(32, 32, 2),
                ir_group(32, 32, 2),
            ),
            nn.Sequential(
                ir5(32, 64, 4, stride=2),
                ir(64, 64, 3),
                ir(64, 64, 3),
                ir(64, 64, 3),
                ir(64, 64, 3),
            ),
            nn.Sequential(
                ir5(64, 96, 5, stride=2),
                ir(96, 96, 3),
                ir(96, 96, 3),
                ir(96, 96, 3),
                ir(96, 96, 3),
                ir(96, 160, 5),
                *(ir(160, 160, 3) for _ in range(8)),
            ),
            nn.Sequential(
                ir(160, 256, 6, stride=2),
                *(ir(256, 256, 5) for _ in range(6)),
                ir(256, 288, 6),
            ),
        ]
        for idx, stage in enumerate(stages):
            self.add_module(f"stage{idx}", stage)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, stage in self.named_children():
            x = stage(x)
            out[name] = x
        return out


# ---------------------------------------------------- Neck & ConvStack Heads

class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="replicate"),
        )
        self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.skip_connection(x)


class _Resampler(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kind: str, scale_factor: int = 2) -> None:
        if kind == "conv_transpose":
            super().__init__(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            )
        elif kind == "bilinear":
            super().__init__(
                nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            )
        else:
            raise ValueError(f"Unsupported resampler kind: {kind!r}")


class _ConvStack(nn.Module):
    def __init__(
        self,
        dim_in: Sequence[int | None],
        dim_res_blocks: Sequence[int],
        dim_out: Sequence[int | None] | None,
        num_res_blocks: Sequence[int],
        resamplers: Sequence[str],
    ) -> None:
        super().__init__()
        self.input_blocks = nn.ModuleList(
            [
                nn.Conv2d(int(d_in), d_res, kernel_size=1, stride=1, padding=0)
                if d_in is not None
                else nn.Identity()
                for d_in, d_res in zip(dim_in, dim_res_blocks)
            ]
        )
        self.resamplers = nn.ModuleList(
            [
                _Resampler(d_prev, d_next, kind=kind, scale_factor=2)
                for d_prev, d_next, kind in zip(dim_res_blocks[:-1], dim_res_blocks[1:], resamplers)
            ]
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(*(_ResidualConvBlock(d_res) for _ in range(n_blocks)))
                for d_res, n_blocks in zip(dim_res_blocks, num_res_blocks)
            ]
        )
        dim_out_seq = dim_out if dim_out is not None else [None] * len(dim_res_blocks)
        self.output_blocks = nn.ModuleList(
            [
                nn.Conv2d(d_res, int(d_out), kernel_size=1, stride=1, padding=0)
                if d_out is not None
                else nn.Identity()
                for d_out, d_res in zip(dim_out_seq, dim_res_blocks)
            ]
        )

    def forward(self, in_features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        out_features: list[torch.Tensor] = []
        x: torch.Tensor | None = None
        for i in range(len(self.res_blocks)):
            feat = self.input_blocks[i](in_features[i])
            x = feat if i == 0 else (x + feat if x is not None else feat)
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features


class _MLPHead(nn.Sequential):
    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__(
            *itertools.chain(
                *((nn.Linear(d_in, d_out), nn.ReLU(inplace=True)) for d_in, d_out in zip(dims[:-2], dims[1:-1]))
            ),
            nn.Linear(dims[-2], dims[-1]),
        )


# --------------------------------------------------------------- HyDenMoGe

class HyDenMoGe(nn.Module):
    """HyDen MoGe-2 metric point and depth estimator (`vitl_dinov2` + `CNNLarge`)."""

    def __init__(
        self,
        *,
        intermediate_layers: Sequence[int] = (5, 11, 17, 23),
        vit_dim_out: int = 1024,
        vit_img_size: int = 518,
        patch_size: int = 14,
        vit_embed_dim: int = 1024,
        vit_depth: int = 24,
        vit_num_heads: int = 16,
        vit_internal_resolution: tuple[int, int] | None = (518, 518),
        dim_res_blocks: Sequence[int] = (1024, 256, 128, 64, 32),
        neck_res_blocks: Sequence[int] = (0, 2, 2, 2, 0),
        head_res_blocks: Sequence[int] = (0, 1, 1, 1, 0),
    ) -> None:
        super().__init__()
        self.default_patch_size = patch_size
        self.vit_internal_resolution = vit_internal_resolution
        resamplers = ("conv_transpose", "conv_transpose", "conv_transpose", "bilinear")

        self.encoder = DINOv2Encoder(
            intermediate_layers=intermediate_layers,
            dim_out=vit_dim_out,
            img_size=vit_img_size,
            patch_size=patch_size,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
        )
        self.cnn_encoder = CNNLarge(in_channels=3)
        self.image_feats = nn.Sequential(
            nn.Conv2d(CNN_TOTAL_CHANNELS + vit_dim_out, CNN_TOTAL_CHANNELS + vit_dim_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_TOTAL_CHANNELS + vit_dim_out, vit_dim_out, kernel_size=3, padding=1),
        )
        self.scale_feats = nn.Sequential(
            nn.Linear(CNN_LAST_CHANNEL + vit_embed_dim, CNN_LAST_CHANNEL + vit_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(CNN_LAST_CHANNEL + vit_embed_dim, vit_dim_out),
        )
        self.neck = _ConvStack(
            dim_in=[vit_dim_out + 2, 2, 2, 2, 2],
            dim_res_blocks=dim_res_blocks,
            dim_out=None,
            num_res_blocks=neck_res_blocks,
            resamplers=resamplers,
        )
        self.points_head = _ConvStack(
            dim_in=dim_res_blocks,
            dim_res_blocks=dim_res_blocks,
            dim_out=[None, None, None, None, 3],
            num_res_blocks=head_res_blocks,
            resamplers=resamplers,
        )
        self.mask_head = _ConvStack(
            dim_in=dim_res_blocks,
            dim_res_blocks=dim_res_blocks,
            dim_out=[None, None, None, None, 1],
            num_res_blocks=head_res_blocks,
            resamplers=resamplers,
        )
        self.scale_head = _MLPHead(dims=[vit_dim_out, vit_dim_out, vit_dim_out, 1])

    @classmethod
    def tiny_for_tests(cls, seed: int = 0) -> "HyDenMoGe":
        """Deterministic tiny CPU instance for fast unit and CLI checks without 1.5 GB weights."""
        torch.manual_seed(seed)
        model = cls(
            intermediate_layers=(0, 1),
            vit_dim_out=32,
            vit_img_size=56,
            patch_size=14,
            vit_embed_dim=32,
            vit_depth=2,
            vit_num_heads=4,
            vit_internal_resolution=(56, 56),
            dim_res_blocks=(32, 16, 16, 8, 8),
            neck_res_blocks=(0, 1, 1, 1, 0),
            head_res_blocks=(0, 1, 1, 1, 0),
        )
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(0.05)
            # Route level-4 UV (channels 0, 1) cleanly into points (X, Y) so focal/shift recovery is well-posed
            neck_in4 = model.neck.input_blocks[4]
            assert isinstance(neck_in4, nn.Conv2d)
            neck_in4.weight.zero_()
            neck_in4.bias.zero_()
            neck_in4.weight[0, 0, 0, 0] = 1.0
            neck_in4.weight[1, 1, 0, 0] = 1.0

            pts_in4 = model.points_head.input_blocks[4]
            pts_out4 = model.points_head.output_blocks[4]
            assert isinstance(pts_in4, nn.Conv2d) and isinstance(pts_out4, nn.Conv2d)
            pts_in4.weight.zero_()
            pts_in4.bias.zero_()
            for c in range(8):
                pts_in4.weight[c, c, 0, 0] = 1.0
            pts_out4.weight.zero_()
            pts_out4.bias.zero_()
            pts_out4.weight[0, 0, 0, 0] = 1.0
            pts_out4.weight[1, 1, 0, 0] = 1.0
            pts_out4.weight[2, 0, 0, 0] = 0.25
            pts_out4.weight[2, 1, 0, 0] = 0.20
            pts_out4.weight[2, 2, 0, 0] = 0.50
            pts_out4.bias[2] = 0.5

            nn.init.constant_(model.mask_head.output_blocks[-1].bias, 4.0)
            nn.init.constant_(model.scale_head[-1].bias, 1.5)
        model.eval()
        return model

    def forward(
        self,
        image: torch.Tensor,
        return_mask_and_scale: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass on raw `[0, 1]` `float32` RGB `(B, 3, H, W)`."""
        validate_raw_rgb(image)
        batch_size, _, orig_h, orig_w = image.shape
        device, dtype = image.device, image.dtype
        image_cnn = image

        if self.vit_internal_resolution is not None:
            image_vit = F.interpolate(
                image,
                size=self.vit_internal_resolution,
                mode="bilinear",
                align_corners=True,
            )
            _, _, vit_h, vit_w = image_vit.shape
        else:
            image_vit = image
            vit_h, vit_w = orig_h, orig_w

        num_patches_h = vit_h // self.default_patch_size
        num_patches_w = vit_w // self.default_patch_size
        num_tokens = num_patches_h * num_patches_w
        aspect_ratio = vit_w / vit_h
        base_h = int((num_tokens / aspect_ratio) ** 0.5)
        base_w = int((num_tokens * aspect_ratio) ** 0.5)

        features, cls_token = self.encoder(image_vit, base_h, base_w)

        cnn_out = self.cnn_encoder(image_cnn)
        num_patches_h_cnn = orig_h // self.default_patch_size
        num_patches_w_cnn = orig_w // self.default_patch_size
        sorted_keys = sorted(cnn_out.keys())
        cnn_features = torch.cat(
            [
                F.interpolate(
                    cnn_out[k],
                    size=(num_patches_h_cnn, num_patches_w_cnn),
                    mode="bilinear",
                    align_corners=False,
                )
                for k in sorted_keys
            ],
            dim=1,
        )

        last_stage = cnn_out[sorted_keys[-1]]
        cls_token_cnn = F.avg_pool2d(last_stage, kernel_size=last_stage.shape[2:]).squeeze(-1).squeeze(-1)
        cls_token = self.scale_feats(torch.cat([cls_token, cls_token_cnn], dim=1))

        features = F.interpolate(
            features,
            size=(num_patches_h_cnn, num_patches_w_cnn),
            mode="bilinear",
            align_corners=False,
        )
        features = self.image_feats(torch.cat([features, cnn_features], dim=1))
        base_h, base_w = num_patches_h_cnn, num_patches_w_cnn

        multi_features: list[torch.Tensor | None] = [features, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2 ** level,
                height=base_h * 2 ** level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if multi_features[level] is None:
                multi_features[level] = uv
            else:
                cur = multi_features[level]
                assert cur is not None
                if cur.shape[2:] != uv.shape[2:]:
                    cur = F.interpolate(cur, size=(uv.shape[2], uv.shape[3]), mode="bilinear", align_corners=False)
                multi_features[level] = torch.cat([cur, uv], dim=1)

        neck_out = self.neck([f for f in multi_features if f is not None])
        raw_points = self.points_head(neck_out)[-1]
        raw_points = F.interpolate(raw_points, (orig_h, orig_w), mode="nearest").permute(0, 2, 3, 1)
        xy, z = raw_points.split([2, 1], dim=-1)
        z = torch.exp(z)
        points = torch.cat([xy * z, z], dim=-1)

        out: dict[str, torch.Tensor] = {"points": points}
        if return_mask_and_scale:
            raw_mask = self.mask_head(neck_out)[-1]
            mask = F.interpolate(raw_mask, (orig_h, orig_w), mode="nearest").squeeze(1).sigmoid()
            metric_scale = self.scale_head(cls_token).squeeze(1)
            out["mask"] = mask
            out["metric_scale"] = metric_scale
        return out

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        *,
        fov_x: float | torch.Tensor | None = None,
        force_projection: bool = True,
        apply_mask: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run forward + focal/shift/intrinsics recovery + metric scale on raw `[0, 1]` `float32` RGB."""
        squeeze = image.ndim == 3
        if squeeze:
            image = image.unsqueeze(0)
        validate_raw_rgb(image)
        orig_h, orig_w = image.shape[-2:]
        aspect_ratio = orig_w / orig_h

        fwd = self.forward(image, return_mask_and_scale=True)
        points = fwd["points"]
        mask_prob = fwd["mask"]
        metric_scale = fwd["metric_scale"]
        mask_binary = mask_prob > 0.5

        points, depth, intrinsics, mask_binary = recover_intrinsics(
            points, mask_binary, fov_x=fov_x, aspect_ratio=aspect_ratio
        )
        if force_projection:
            points = depth_to_points(depth, intrinsics=intrinsics)

        points = points * metric_scale[:, None, None, None]
        depth = depth * metric_scale[:, None, None]

        if apply_mask and mask_binary is not None:
            points = torch.where(mask_binary[..., None], points, torch.inf)
            depth = torch.where(mask_binary, depth, torch.inf)

        res: dict[str, torch.Tensor] = {
            "points": points,
            "depth": depth,
            "intrinsics": intrinsics,
            "mask": mask_binary,
            "mask_prob": mask_prob,
            "metric_scale": metric_scale,
        }
        if squeeze:
            res = {k: v.squeeze(0) for k, v in res.items()}
        return res


def load_hyden_moge2(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cpu",
    vit_internal_resolution: tuple[int, int] | None = (518, 518),
) -> HyDenMoGe:
    """Load `HyDenMoGe` from `checkpoint_path` with `strict=True` (all 953 keys required)."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"HyDen MoGe-2 checkpoint not found: {path}")
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    model = HyDenMoGe(vit_internal_resolution=vit_internal_resolution)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model
