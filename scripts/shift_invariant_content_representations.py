import torch
import numpy as np
import matplotlib as mpl
import seaborn as sns
import rootutils

from pathlib import Path
from torch.functional import F
from matplotlib import pyplot as plt
from torchvision import transforms as T
from sklearn.mixture import GaussianMixture

from src.core.transforms.translation import translation
from src.core.data.sounding_out_chorus import SoundingOutChorus
from src.core.utils.sketch import plot_mel_spectrogram

plt.rcParams.update({
    'font.size': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

def designal(x):
    x_bg = copy.deepcopy(x)
    for i in range(x_bg.shape[1]):
        f = x[:, i]
        gm = GaussianMixture(n_components=2).fit(f.reshape(-1, 1))
        f_max = gm.means_.min()
        bg_mask = f < f_max
        ts, = np.where(f > f_max)
        for t in ts:
            x_bg[t, i] = np.random.choice(f[bg_mask]).item()
    return x_bg

def central_finite_difference(x, padding_mode="reflect"):
    x = F.pad(x.unsqueeze(-2), (1, 1), padding_mode)
    kernel = torch.tensor([[[-1.0, 0.0, 1.0]]]).to(x.device)
    dxdt = F.conv1d(x, kernel).squeeze(-2)
    return dxdt

def main(
    data_path = "/mnt/data0/kag25/sounding_out",
) -> None:
    width = 20
    height = 2.4*3
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid_spec = fig.add_gridspec(
        nrows=4, ncols=10,
        width_ratios=[*[1/9 for i in range(9)], 0.02],
        height_ratios=[0.32, 0.04, 0.32, 0.32],
    )

    base = get_vae()
    nifti = get_vae()
    transforms = get_transforms()
    data = get_data(transforms)

    coords_ax = fig.add_subplot(grid_spec[1, :-1])
    colours = [plt.get_cmap('twilight_shifted')(1.*i/255) for i in range(256)]
    ts = np.linspace(-1, 1, 256)
    positions = [0, 31, 63, 95, 127, 159, 191, 223, 255]
    # coords_ax_2 = coords_ax.twiny()
    gradients = np.vstack((ts, ts))
    coords_ax.imshow(gradients, aspect='auto', cmap="twilight_shifted")
    coords_ax.tick_params(labelleft=False, left=False)
    coords_ax.set_xticks(positions, labels=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    # coords_ax_2.set_xticks(positions, labels=[np.format_float_positional(x, precision=3) for x in np.arange(9) * 1.536 / (9 - 1)])

    x = transforms(data.load_sample("KN-10_1_20150508_0600.wav")).squeeze()
    samples_per_second = int(np.ceil(LOG_MEL_SPECTROGRAM_PARAMS["sample_rate"] / LOG_MEL_SPECTROGRAM_PARAMS["hop_length"]))
    t_i, t_j, f_i, f_j = 2530, 2564, 20, 30
    signal = x[t_i:t_j, f_i:f_j]
    s = signal.size(0)
    
    x_bg = torch.tensor(designal(x.numpy()))[:192]
    t_start = (x_bg.size(0)) // 2
    x_bg[t_start - s:t_start, f_i:f_j] = signal

    xs = []
    dts = np.linspace(-1.0, 1.0, 9)
    for i, dt in enumerate(dts):
        xs.append(translation(x_bg.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, 1, 1) * dt, padding_mode="circular").squeeze())

    for i, (x_bg, dt) in enumerate(zip(xs, dts)):
        ax = fig.add_subplot(grid_spec[0, i])
        twin_ax = ax.twiny()
        im = plot_mel_spectrogram(20 * np.log10(x_bg.exp().numpy().T), **LOG_MEL_SPECTROGRAM_PARAMS, cmap="Greys", ax=ax)
        ax.set_xticks([0, 95, 191], [-1.0, 0.0, 1.0])
        twin_ax.set_xticks([0, 95, 191], [0.0, 1.536/2, 1.536])
        # ax.set_title(rf"$\Delta t = {{{np.format_float_positional(dt, precision=2, min_digits=2)}}}$")
        ax.set_xlabel("")
        ax.set_ylabel("")
        # ax.axvline(x=95, linestyle="dashed", color="black", linewidth=3.0)
        colour = colours[positions[i]]
        print(mpl.colors.to_hex(colour))
        # ax.axvline(x=[0, 23, 47, 71, 95, 119, 143, 167, 191][i], linestyle="dashed", color=colour, linewidth=3.0) 
        if i != 0:
            ax.tick_params(labelleft=False, left=False)

    dts = np.linspace(-1, 1, 105)
    batch = torch.cat([translation(x_bg.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, 1, 1) * dt, padding_mode="circular") for dt in dts], dim=0)
    
    base_q_z, *_ = base.encode(batch.detach())
    base_mu_x = base_q_z.detach().chunk(2, dim=-1)[0].squeeze()
    base_dzdT = central_finite_difference(base_mu_x.t(), padding_mode="circular")

    nifti_q_z, *_ = nifti.encode(batch.detach())
    nifti_mu_x = nifti_q_z.detach().chunk(2, dim=-1)[0].squeeze()
    nifti_dzdT = central_finite_difference(nifti_mu_x.t(), padding_mode="circular")
    
    vmin, vmax = min(base_dzdT.min(), nifti_dzdT.min()), max(base_dzdT.max(), nifti_dzdT.max())
    imshow_params = dict(origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=sns.color_palette("vlag", as_cmap=True))
    
    ax = fig.add_subplot(grid_spec[2, :-1])
    im = ax.imshow(base_dzdT, **imshow_params)
    ax.tick_params(labelbottom=False, bottom=False)
    ax.set_yticks(np.arange(0, 128 + 16, 16), np.arange(0, 128 + 16, 16))

    for p in np.arange(0, 105, 13):
        ax.axvline(x=p + 0.5, linestyle="dashed", color="black", linewidth=1.0)
        ax.axvline(x=p - 0.5, linestyle="dashed", color="black", linewidth=1.0)

    ax = fig.add_subplot(grid_spec[3, :-1])
    im = ax.imshow(nifti_dzdT, **imshow_params)
    ax.set_yticks(np.arange(0, 128 + 16, 16), np.arange(0, 128 + 16, 16))
    ax.set_xticks(np.arange(0, 105, 13), [dts[i] for i in np.arange(0, 105, 13)])

    for p in np.arange(0, 105, 13):
        ax.axvline(x=p + 0.5, linestyle="dashed", color="black", linewidth=1.0)
        ax.axvline(x=p - 0.5, linestyle="dashed", color="black", linewidth=1.0)

    cbar = fig.colorbar(im, cax=fig.add_subplot(grid_spec[2:, -1]))
    plt.show()

main()
