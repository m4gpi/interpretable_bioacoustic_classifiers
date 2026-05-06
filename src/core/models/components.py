import lightning as L
import torch
import functools
from dataclasses import dataclass
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from torch import nn
from torch import Tensor
from typing import Any, List, Tuple

__all__ = [
    "Activation",
    "NormType",
    "ResidualConv2d",
]

class ArcTan2(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("identity", torch.tensor([1.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.identity
        dx, dy = x.chunk(2, dim=-1)
        norm = (dx.pow(2) + dy.pow(2) + self.epsilon).sqrt().detach()
        dx, dy = dx / norm, dy / norm
        theta = torch.atan2(dy, dx)
        return theta, dx, dy

class Activation(Enum):
    NULL = (nn.Identity,)
    RELU = (nn.ReLU,)
    GELU = (nn.GELU,)
    SELU = (nn.SELU,)
    SILU = (nn.SiLU,)
    LEAK = (nn.LeakyReLU,)
    ATAN2 = (ArcTan2,)

    def __init__(self, init: Callable[[], nn.Module]) -> None:
        self.init = init

class NormType(Enum):
    BN1 = (nn.BatchNorm1d,)
    BN2 = (nn.BatchNorm2d,)
    LN = (functools.partial(nn.GroupNorm, 1),)
    GN = (nn.GroupNorm,)

    def __init__(self, init: Callable[[int], nn.Module]) -> None:
        self.init = init

class Resample(Enum):
    NULL = (nn.Identity,)
    MAX = (functools.partial(nn.MaxPool2d, kernel_size=2, stride=2),)
    AVG = (functools.partial(nn.AvgPool2d, kernel_size=2, stride=2),)
    CONV_DOWN = (functools.partial(nn.Conv2d, kernel_size=2, stride=2),)
    CONV_UP = (functools.partial(nn.ConvTranspose2d, kernel_size=2, stride=2),)
    LIN_UP = (torch.nn.Upsample,)

    def __init__(self, init: Callable[[], nn.Module]) -> None:
        self.init = init

class ResidualConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Activation,
        norm: NormType,
        dropout_prob: float = 0.5,
        padding_mode: str = 'zeros',
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
            norm.init(out_channels),
            activation.init(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm.init(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + self.shortcut(x)

def init_cnn_feature_encoder(
    block_sizes: List[int],
    block_width: int,
    block_depth: int,
    dropout_prob: float,
    padding_mode: str,
    norm_fn: NormType,
    activation_fn: Activation,
    weight_init_std: float,
) -> nn.Module:
    in_channels = block_sizes[0] * block_width
    encoder_cnn = nn.ModuleList()
    pre_process = nn.Sequential(
        nn.Conv2d(1, in_channels, kernel_size=5, padding=2, padding_mode=padding_mode),
        norm_fn.init(in_channels),
        activation_fn.init(),
        nn.Conv2d(in_channels, in_channels * 2, kernel_size=(2, 1), stride=(2, 1), padding_mode=padding_mode),
        nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
    )
    nn.init.trunc_normal_(pre_process[0].weight, std=weight_init_std)
    encoder_cnn.append(pre_process)
    in_channels = block_sizes[0] * block_width
    for block_size in block_sizes[1:]:
        block = nn.Sequential()
        out_channels = block_size * block_width
        down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        block.append(down)
        block.extend([ResidualConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=activation_fn,
            norm=norm_fn,
            dropout_prob=dropout_prob,
            padding_mode=padding_mode,
        ) for i in range(block_depth)])
        encoder_cnn.append(block)
        in_channels = out_channels
    return encoder_cnn

def init_cnn_feature_decoder(
    block_sizes: List[int],
    block_width: int,
    block_depth: int,
    dropout_prob: float,
    padding_mode: str,
    norm_fn: NormType,
    activation_fn: Activation,
) -> nn.Module:
    decoder_cnn = nn.ModuleList()
    in_channels = block_sizes[0] * block_width
    for i, block_size in enumerate(block_sizes[1:]):
        block = nn.Sequential()
        out_channels = block_size * block_width
        block.extend([ResidualConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            activation=activation_fn,
            norm=norm_fn,
            dropout_prob=dropout_prob,
            padding_mode=padding_mode,
        ) for j in range(block_depth)])
        up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        block.append(up)
        decoder_cnn.append(block)
        in_channels = out_channels
    # final expansion in time dimension
    decoder_cnn.append(nn.Sequential(
        nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
        norm_fn.init(in_channels * 2),
        activation_fn.init(),
        nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=(2, 1), stride=(2, 1)),
        norm_fn.init(in_channels),
        activation_fn.init(),
        nn.Conv2d(in_channels, 1, kernel_size=5, padding=2, padding_mode=padding_mode),
    ))
    return decoder_cnn

def init_mlp_content_encoder(
    in_channels: int,
    out_channels: int,
    feature_height: int,
    feature_width: int,
    mlp_reduction_factor: float,
    activation_fn: Activation,
    dropout_prob: float,
    out_features: int,
) -> nn.Module:
    in_features = out_channels * feature_height * feature_width
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.Flatten(),
        nn.Linear(in_features=in_features, out_features=in_features // mlp_reduction_factor),
        activation_fn.init(),
        nn.Dropout(p=dropout_prob),
        nn.Linear(in_features=in_features // mlp_reduction_factor, out_features=out_features)
    )

def init_mlp_content_decoder(
    in_features: int,
    in_channels: int,
    out_channels: int,
    feature_height: int,
    feature_width: int,
    mlp_reduction_factor: float,
    activation_fn: Activation,
    dropout_prob: float,
) -> nn.Module:
    out_features = in_channels * feature_width * feature_height
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features // mlp_reduction_factor),
        activation_fn.init(),
        nn.Dropout(p=dropout_prob),
        nn.Linear(in_features=out_features // mlp_reduction_factor, out_features=out_features),
        nn.Unflatten(-1, (in_channels, feature_height, feature_width)),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    )

def init_alignment_encoder(
    in_channels: int,
    out_channels: int,
    cnn_kernel_size: Tuple[int, int],
    flatten_start_dim: int,
    in_features: int,
    activation_fn: Activation,
    mlp_reduction_factor: int,
    out_features: int,
) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=cnn_kernel_size),
        nn.Flatten(start_dim=flatten_start_dim),
        nn.Linear(in_features=in_features, out_features=in_features // mlp_reduction_factor),
        activation_fn.init(),
        nn.Linear(in_features=in_features // mlp_reduction_factor, out_features=out_features)
    )

class ResidualCNN(torch.nn.Module):
    def __init__(
        self,
        block_sizes: List[int],
        block_width: int,
        block_depth: int,
        block_resample: List[str],
        dropout_prob: float,
        padding_mode: str,
        norm_type: NormType,
        activation: Activation,
    ) -> None:
        super().__init__()
        self.block_sizes = block_sizes
        self.block_width = block_width
        self.block_depth = block_depth
        self.block_resample = block_resample
        self.padding_mode = padding_mode
        self.norm_type = norm_type
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.num_layers = len(self.block_sizes)

        norm_fn = NormType[norm_type]
        activation_fn = Activation[activation]
        resample_fns = [Resample[op] for op in block_resample]

        self.blocks = nn.ModuleList()
        in_channels = block_sizes[0] * block_width
        for i, (block_size, resample) in enumerate(zip(block_sizes[1:], resample_fns)):
            block = nn.Sequential()
            out_channels = block_size * block_width
            if resample.name == "CONV_DOWN":
                block.append(resample.init(in_channels=in_channels, out_channels=out_channels))
                in_channels = out_channels
            block.extend([
                ResidualConv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    activation=activation_fn,
                    norm=norm_fn,
                    dropout_prob=dropout_prob,
                    padding_mode=padding_mode,
                )
                for j in range(block_depth)
            ])
            if resample.name == "CONV_UP":
                block.append(resample.init(in_channels=in_channels, out_channels=out_channels))
                in_channels = out_channels
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
        return x, features

    @property
    def rescaling_factor(self):
        return len([mode for mode in self.block_resample if mode != "NULL"])

class MLP(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str,
        dropout_prob: float,
    ) -> None:
        assert len(layer_sizes) >= 3, "An MLP must have at least one hidden layer"
        super().__init__()
        self.activation = activation
        self.dropout_prob = dropout_prob
        activation_fn = Activation[activation]
        layers = []
        layers.append(torch.nn.Linear(layer_sizes[0], layer_sizes[1]))
        for i in range(1, len(layer_sizes) - 1):
            layers.append(activation_fn.init())
            layers.append(torch.nn.Dropout(p=dropout_prob)),
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CNNFeatureEncoder(ResidualCNN):
    def __init__(
        self,
        in_channels: int,
        *args: Any,
        entrypoint_weight_std: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        out_channels = self.block_sizes[0] * self.block_width
        pre_process = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, padding_mode=self.padding_mode),
            NormType[self.norm_type].init(out_channels),
            Activation[self.activation].init(),
            torch.nn.Conv2d(out_channels, out_channels * 2, kernel_size=(2, 1), stride=(2, 1), padding_mode=self.padding_mode),
            torch.nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
        )
        torch.nn.init.trunc_normal_(pre_process[0].weight, std=entrypoint_weight_std)
        self.blocks = torch.nn.ModuleList([pre_process, *self.blocks])

class ContentEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_sizes: List[int],
        activation: str,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.conv_down = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.mlp = MLP(layer_sizes, activation, dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_down(x)
        x = x.flatten(start_dim=-3)
        return self.mlp(x)

class ContentDecoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_height: int,
        out_width: int,
        layer_sizes: List[int],
        activation: str,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        assert layer_sizes[-1] == in_channels * out_height * out_width, \
            f"MLP output dimension ({layer_sizes[-1]}) must match conv ({in_channels} * {out_height} * {out_width})"
        self.in_channels = in_channels
        self.out_height = out_height
        self.out_width = out_width
        self.mlp = MLP(layer_sizes, activation, dropout_prob)
        self.conv_up = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = x.unflatten(-1, (self.in_channels, self.out_height, self.out_width))
        x = self.conv_up(x)
        return x

class CNNFeatureDecoder(ResidualCNN):
    def __init__(
        self,
        out_channels: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        in_channels = self.block_sizes[-1] * self.block_width
        post_process = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            NormType[self.norm_type].init(in_channels * 2),
            Activation[self.activation].init(),
            torch.nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=(2, 1), stride=(2, 1)),
            NormType[self.norm_type].init(in_channels),
            Activation[self.activation].init(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, padding_mode=self.padding_mode),
        )
        self.blocks.append(post_process)

class XAlignmentEncoder(torch.nn.Module):
    def __init__(
        self,
        out_features: int = 1,
        x_channels: int = 512,
        x_freq_dim: int = 4,
        x_time_dim: int = 6,
        weight_init_std: float = 1e-1,
        activation: str = "NONE",
    ) -> None:
        super().__init__()
        self.x_conf_freq = torch.nn.Conv2d(x_channels, x_channels // 4, kernel_size=(1, x_freq_dim))
        in_features = x_channels // 4 * x_time_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features // 2, out_features, bias=False),
            Activation[activation].init(),
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        bs, seq, ch, ts, fq = x.shape
        x = x.flatten(end_dim=1)
        x = self.x_conf_freq(x).squeeze(-1)
        x = x.unflatten(0, (bs, seq)).flatten(start_dim=-2)
        return self.mlp(x)

class UXAlignmentEncoder(nn.Module):
    def __init__(
        self,
        out_features: int = 1,
        proj_dim: int = 512,
        x_channels: int = 512,
        x_freq_dim: int = 4,
        x_time_dim: int = 48,
        u_channels: int = 64,
        u_freq_dim: int = 32,
        u_time_dim: int = 6,
        u_weight_init_std: float = 1e-1,
        activation: str = "NONE",
    ) -> None:
        super().__init__()
        self.x_conv_freq = torch.nn.Conv2d(x_channels, x_channels // 4, kernel_size=(1, x_freq_dim))
        self.u_conv_freq = torch.nn.Conv2d(u_channels, u_channels // 4, kernel_size=(1, u_freq_dim))
        self.x_proj = torch.nn.Linear(x_channels // 4 * x_time_dim, proj_dim)
        self.u_proj = torch.nn.Linear(u_channels // 4 * u_time_dim, proj_dim)
        self.u_proj.weight.data *= u_weight_init_std # reduce noise coming from u early in training
        in_features = proj_dim * 2
        self.mlp = torch.nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features // 2, out_features, bias=False),
            Activation[activation].init(),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: int | None = None):
        bs, seq, ch, ts, fq = x.shape
        # deep features
        x = x.flatten(end_dim=1)
        x = self.x_conv_freq(x).squeeze(-1)
        x = x.unflatten(0, (bs, seq)).flatten(start_dim=-2)
        # shallow features
        u = u.flatten(end_dim=1)
        u = self.u_conv_freq(u).squeeze(-1)
        u = u.unflatten(0, (bs, seq)).flatten(start_dim=-2)
        # combine through projection and concatenation
        x = self.x_proj(x) # (bs, seq, d)
        u = self.u_proj(u) # (bs, seq, d)
        h = torch.cat([x, u], dim=-1)
        # predict alignment factor
        return self.mlp(h)
