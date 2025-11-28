import torch
import numpy as np

from torch import Tensor
from torch import nn
from typing import Optional, Any
from torch.functional import F

from src.core.utils import try_or

__all__ = ["frame_fold", "unframe_fold", "frame_fast", "unframe_fast", "Frame", "Unframe"]

def frame_fold(x: Tensor, hop_length: int, window_length: int, padding_mode: Optional[str] = None) -> Tensor:
    """
    Compute an ordered series of frames across the height of a 3D Tensor

    :param Tensor x: shape (N, C, H, W)
    :param int hop_length: the stride across the height axis
    :param int window_length: the receptive field for a given stride
    :param str|none padding_mode: optional parameter to specify padding method, see torch.functional.pad
    :returns Tensor x: shape (N, T, C, window_length, W)
    """
    # add batch & channel dimensions
    if len(x.size()) <= 3:
        ch = try_or(lambda: x.size(-3), 1)
        x = x.view(1, ch, x.size(-2), x.size(-1))
    N, C, H, W = x.size()
    T = int(np.floor(((H - (window_length - 1) - 1) / hop_length) + 1))
    if padding_mode and (remaining := x.size(-2) % hop_length):
        # when the window is greater than the hop, pad at the borders
        pad = window_length - remaining
        # NB: presupposes that pad is equally divisible by 2
        x = F.pad(x.transpose(-1, -2), (pad // 2, pad // 2, 0, 0), padding_mode).transpose(-1, -2)
    # apply framing in time to compute a sequence of windows
    unfold = nn.Unfold(kernel_size=(window_length, W), stride=(hop_length, 1))
    # unfold to output (N, C * W * window_lenth, T)
    x_framed = unfold(x)
    # reshape to (N, T, C, window, W)
    x_framed = x_framed.transpose(-1, -2).reshape(N, T, C, window_length, W)
    return x_framed

def unframe_fold(x: Tensor, hop_length: int, num_timesteps: Optional[int]) -> Tensor:
    """
    Undo a frame operation to stack back together an ordered series of frames into a 3D Tensor

    :param Tensor x: shape (N, T, C, window_length, W)
    :param int hop_length: the stride across the height axis
    :param int num_timesteps: the output size of the unframed dimension (to account for padding in frame operation)
    :returns Tensor x: shape (N, C, H, W)
    """
    T, C, window_length, W = x.size()[-4:]
    x = x.flatten(start_dim=-3).transpose(-1, -2)
    # retreive the original (possible padded) height based on the size of the series and the hop and window lengths
    H = (T * hop_length) + (window_length - hop_length)
    fold = nn.Fold(output_size=(H, W), kernel_size=(window_length, W), stride=(hop_length, 1))
    unfold = nn.Unfold(kernel_size=(window_length, W), stride=(hop_length, 1))
    # when the window was greater than the hop, fold sums the values, so we take the average
    divisor = torch.ones(1, C, H, W).to(x.device)
    x = fold(x) / fold(unfold(divisor))
    # remove padding (possibly added in the forward frame operation)
    pad = (H - (num_timesteps or H))
    if pad > 0:
        # NB: presupposes that pad is equally divisible by 2
        x = x[:, :, pad//2:-pad//2]
    return x

def frame_fast(
    x: Tensor,
    window_length: int,
    hop_length: int,
) -> Tensor:
    """
    Expand an nD tensor to an nD+1 tensor using a windowing function

    - Calculate the number of frames according to the window, ensuring
      the tail is removed when the size of the frames doesn't match
    - Size switches the old dimension, 'num_samples', to two new dimensions,
      'num_frames' and 'window_length'
    - From existing strides, to get to the next frame, step by the stride
      times the hop length, keep existing strides
    """
    num_samples, *other_dims = x.shape
    num_frames = 1 + int(round((num_samples - window_length) // hop_length))
    return x.as_strided(
        size=(num_frames, window_length, *other_dims),
        stride=(x.stride()[0] * hop_length, *x.stride())
    )

def unframe_fast(
    x: Tensor,
    window_length: int,
    hop_length: int,
) -> Tensor:
    """
    Reduce an nD tensor to an nD-1 tensor using a windowing function

    - Original number of samples is unavailable, we cannot know how many
      were discarded (within this scope), use what we know to remove the overlap
    - Calculate the difference between the window length and the hop length 
      times the number of frames - 1
    """
    num_frames, window_length, *other_dims = x.shape
    num_samples = (num_frames * window_length) - int((num_frames - 1) * (window_length - hop_length))
    return x.as_strided(
        size=(num_samples, *other_dims),
        stride=x.stride()[1:]
    )

class Frame:
    def __init__(
        self,
        window_length: int,
        hop_length: int,
        padding_mode: Optional[str] = None,
    ) -> None:
        self.window_length = window_length
        self.hop_length = hop_length
        self.padding_mode = padding_mode

    def __call__(self, x: Tensor) -> Tensor:
        return frame(x, window_length=self.window_length, hop_length=self.hop_length, padding_mode=self.padding_mode)

class Unframe:
    def __init__(
        self,
        hop_length: int,
        num_timesteps: int,
    ) -> None:
        self.hop_length = hop_length
        self.num_timesteps = num_timesteps

    def __call__(self, x: Tensor) -> Tensor:
        return unframe(x, hop_length=self.hop_length, num_timesteps=num_timesteps)
