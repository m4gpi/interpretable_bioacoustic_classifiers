import pytest
import torch
from torch.nn import functional as F

from src.core.models.sivae import SIVAE
from src.core.transforms.translation import translation

@pytest.fixture()
def batch_size():
    return 6

@pytest.fixture()
def x(batch_size):
    return torch.arange(192 * 39).expand(batch_size, 1, 64, -1).transpose(-1, -2)

@pytest.fixture()
def x_framed(x):
    x_framed = x.view(x.size(0), -1, 1, 192, x.size(-1))
    assert (x_framed[0, 0, 0, 0] == torch.ones(x.size(-1)) * 0).all()
    assert (x_framed[0, 0, 0, -1] == torch.ones(x.size(-1)) * 192 - 1).all()
    assert (x_framed[0, -1, 0, 0] == torch.ones(x.size(-1)) * 7296).all()
    assert (x_framed[0, -1, 0, -1] == torch.ones(x.size(-1)) * 7488 - 1).all()
    return x_framed

@pytest.fixture()
def U(batch_size):
    return torch.arange(48 * 39).expand(batch_size, 64, 32, -1).transpose(-1, -2)

@pytest.fixture()
def U_framed(U):
    U_framed = U.view(U.size(0), U.size(1), 39, 48, U.size(-1)).permute(0, 2, 1, 3, 4)
    assert (U_framed[0, 0, 0, 0] == torch.ones(U.size(-1)) * 0).all()
    assert (U_framed[0, 0, 0, -1] == torch.ones(U.size(-1)) * 48 - 1).all()
    assert (U_framed[0, -1, 0, 0] == torch.ones(U.size(-1)) * 1824).all()
    assert (U_framed[0, -1, 0, -1] == torch.ones(U.size(-1)) * 1872 - 1).all()
    return U_framed

def test_spectrogram_framing(x, x_framed):
    model = SIVAE()
    actual = model.frame(x, window_length=192, hop_length=192)
    assert torch.equal(x_framed, actual)

def test_k_way_translations(x_framed, batch_size):
    model = SIVAE()
    k = 2
    seq_len = x_framed.size(1)
    # test consistency at zero
    x_framed = x_framed.float()
    delta = torch.zeros(k - 1, batch_size, seq_len, 1, 1, 1)
    x_trans = model.k_way_translated_frames(x_framed, delta, k - 1, mode="bilinear")
    err = (x_trans - x_framed.unsqueeze(0)).abs().flatten(end_dim=3)
    assert err[:, 1:-1].mean() < 1e-5
    assert (err[:, -1] == err.size(-2) - 1).all()  # modulo introduces a boundary artefact
    # test symmetry
    shift = 0.1
    x_framed = torch.linspace(-1, 1, x_framed.size(-2)).expand(6, 39, 1, 64, -1).transpose(-1, -2)
    delta_1 = torch.ones(k - 1, batch_size, seq_len, 1, 1, 1) * shift
    delta_2 = torch.ones(k - 1, batch_size, seq_len, 1, 1, 1) * -shift
    x_trans_1 = model.k_way_translated_frames(x_framed.float(), delta_1, k - 1, mode="bilinear")
    x_trans_2 = model.k_way_translated_frames(x_framed.float(), delta_2, k - 1, mode="bilinear")
    i = int(shift * x_framed.size(-2)) # ignore regions affected by boundary
    diff_1 = (x_trans_1 - x_framed.unsqueeze(0)).abs().flatten(end_dim=3)[:, i:-i]
    diff_2 = (x_trans_2 - x_framed.unsqueeze(0)).abs().flatten(end_dim=3)[:, i:-i]
    err_1 = diff_1.mean()
    err_2 = diff_2.mean()
    # test magnitude symmetry
    assert torch.allclose(err_1, err_2, atol=1e-4)
    # test spatial symmetry, small diff relative to scale
    err = (diff_1 - diff_2).abs().mean()
    scale = (diff_1.mean() + diff_2.mean()) / 2
    assert err < 0.1 * scale
    # test circularity
    shift = 1.0
    x_framed = torch.linspace(-1, 1, x_framed.size(-2)).expand(6, 39, 1, 64, -1).transpose(-1, -2)
    delta_1 = torch.ones(k - 1, batch_size, seq_len, 1, 1, 1) * shift
    delta_2 = torch.ones(k - 1, batch_size, seq_len, 1, 1, 1) * -shift
    x_trans_1 = model.k_way_translated_frames(x_framed.float(), delta_1, k - 1, mode="bilinear")
    x_trans_2 = model.k_way_translated_frames(x_framed.float(), delta_2, k - 1, mode="bilinear")
    assert torch.allclose(x_trans_1, x_trans_2, atol=1e-6)

def test_decoder_translations(U_framed, batch_size):
    model = SIVAE()
    k = 2
    seq_len = U_framed.size(1)
    U_framed = U_framed.expand(k - 1, -1, -1, -1, -1, -1)

    delta = torch.zeros(k - 1, batch_size, seq_len, 1, 1, 1, 1).float()
    U_trans = model.translate_cnn_features(U_framed.flatten(end_dim=2).float(), delta.flatten(end_dim=2).float(), mode="bilinear")
    U_trans = U_trans.unflatten(0, (U_framed.size(0), U_framed.size(1), U_framed.size(2)))
    err = (U_trans - U_framed).abs().flatten(end_dim=3)
    assert err[:, 1:-1].mean() < 1e-5
    assert (err[:, -1] == err.size(-2) - 1).all()  # modulo introduces a boundary artefact

    # test symmetry
    T = torch.linspace(-1, 1, U_framed.size(-2))
    U_framed = T.expand(U_framed.size(0), U_framed.size(1), U_framed.size(2), U_framed.size(3), U_framed.size(5), -1).transpose(-1, -2)
    shift = 0.1
    delta_1 = torch.ones(k - 1, batch_size, seq_len, 1, 1, 1, 1) * shift
    delta_2 = torch.ones(k - 1, batch_size, seq_len, 1, 1, 1, 1) * -shift
    U_trans_1 = model.translate_cnn_features(U_framed.flatten(end_dim=2).float(), delta_1.flatten(end_dim=2).float(), mode="bilinear")
    U_trans_2 = model.translate_cnn_features(U_framed.flatten(end_dim=2).float(), delta_2.flatten(end_dim=2).float(), mode="bilinear")
    U_trans_1 = U_trans_1.unflatten(0, (U_framed.size(0), U_framed.size(1), U_framed.size(2)))
    U_trans_2 = U_trans_2.unflatten(0, (U_framed.size(0), U_framed.size(1), U_framed.size(2)))
    i = int(shift * U_framed.size(-2)) # ignore regions affected by boundary
    diff_1 = (U_trans_1 - U_framed).abs().flatten(end_dim=3)[:, i:-i]
    diff_2 = (U_trans_2 - U_framed).abs().flatten(end_dim=3)[:, i:-i]
    err_1 = diff_1.mean()
    err_2 = diff_2.mean()
    # test magnitude symmetry
    assert torch.allclose(err_1, err_2, atol=1e-4)
    # test spatial symmetry, small diff relative to scale
    err = (diff_1 - diff_2).abs().mean()
    scale = (diff_1.mean() + diff_2.mean()) / 2
    assert err < 0.1 * scale
    # test circularity
    shift = 1.0
    delta_1 = torch.ones(k - 1, batch_size, seq_len, 1, 1, 1, 1) * shift
    delta_2 = torch.ones(k - 1, batch_size, seq_len, 1, 1, 1, 1) * -shift
    U_trans_1 = model.translate_cnn_features(U_framed.flatten(end_dim=2).float(), delta_1.flatten(end_dim=2).float(), mode="bilinear")
    U_trans_2 = model.translate_cnn_features(U_framed.flatten(end_dim=2).float(), delta_2.flatten(end_dim=2).float(), mode="bilinear")
    U_trans_1 = U_trans_1.unflatten(0, (U_framed.size(0), U_framed.size(1), U_framed.size(2)))
    U_trans_2 = U_trans_2.unflatten(0, (U_framed.size(0), U_framed.size(1), U_framed.size(2)))
    assert torch.allclose(U_trans_1, U_trans_2, atol=1e-6)
