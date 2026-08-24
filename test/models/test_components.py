import torch

from src.core.models.components import CircularReflectiveConv2d

def test_pad():
    stride = 1
    kernel_size = (5, 5)
    # generate some dummy data
    x = torch.arange(0, 16, dtype=torch.float32).reshape(1, 4, 4)
    # test class, set weights and biases to 1/0 for testing purposes
    conv = CircularReflectiveConv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride)
    conv.weight.data = torch.ones(1, 1, *kernel_size, dtype=torch.float32)
    conv.bias.data = torch.tensor([0], dtype=torch.float32)
    actual = conv(x)
    # calculate real expected values
    expected = torch.zeros_like(x)
    k_i, k_j = kernel_size
    p_i, p_j = (k_i - 1) // 2, (k_j - 1) // 2
    x = torch.cat([x[:, :, -p_i:], x, x[:, :, :p_i]], dim=-1)
    x = torch.cat([x[:, 1:p_j+1].flip(dims=[-2]), x, x[:, -p_j-1:-1].flip(dims=[-2])], dim=-2)
    for i in range(p_i, x.size(1) - p_i, stride):
        for j in range(p_j, x.size(2) - p_j, stride):
            expected[:, i-p_i, j-p_i] = (x[:, i-p_i:i+p_i+1, j-p_j:j+p_j+1] * conv.weight.data).sum() + conv.bias.data
    torch.testing.assert_close(expected, actual)
