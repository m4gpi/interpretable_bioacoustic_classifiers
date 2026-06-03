import copy
import lightning as L
import logging
import sklearn
import matplotlib as mpl
import numpy as np
import pathlib
import seaborn as sns
import torch

from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torchvision.transforms import functional as T
from torch.functional import F
from typing import Any, Callable

from src.core.models.vae import VAE
from src.core.models.sivae import SIVAE
from src.core.transforms.translation import translation_1d as translation
from src.core.evaluators.base import Evaluator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

__all__ = ["TranslationInvarianceFigure"]

def central_finite_difference(x, padding_mode="circular"):
    x = F.pad(x, (1, 1), padding_mode)
    kernel = torch.tensor([[[-1.0, 0.0, 1.0]]]).to(x.device)
    dxdt = F.conv1d(x.unsqueeze(-2), kernel).squeeze(-2)
    return dxdt

def designal(x, seed: int = 42):
    np.random.seed(seed)
    x_bg = copy.deepcopy(x)
    for i in range(x_bg.shape[1]):
        f = x[:, i]
        gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=42).fit(f.reshape(-1, 1))
        f_max = gm.means_.min()
        bg_mask = f < f_max
        ts, = np.where(f > f_max)
        for t in ts:
            x_bg[t, i] = np.random.choice(f[bg_mask]).item()
    return x_bg

class TranslationInvarianceFigure(Evaluator):
    def __init__(self, save_path: str, vae_path: str, sivae_path: str):
        self.vae_path = vae_path
        self.sivae_path = sivae_path
        self.save_path = pathlib.Path(save_path)

    def __call__(self, trainer: None, model: Callable, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        L.seed_everything(42)
        log.info(f"loading VAE checkpoint from {self.vae_path}")
        vae = VAE.load_from_checkpoint(self.vae_path, map_location="cpu").eval()
        log.info(f"loading SIVAE checkpoint from {self.sivae_path}")
        sivae = SIVAE.load_from_checkpoint(self.sivae_path, map_location="cpu").eval()

        data = data_module.data
        metadata = data.metadata
        file_name = "KN-10_1_20150508_0600.wav"
        record = metadata[metadata.file_name == file_name].iloc[0]
        file_path = record.file_path

        width = 20
        height = 2.4*3
        fig = plt.figure(figsize=(width, height), constrained_layout=True)
        grid_spec = fig.add_gridspec(
            nrows=4, ncols=10,
            width_ratios=[*[1/9 for i in range(9)], 0.02],
            height_ratios=[0.32, 0.04, 0.32, 0.32],
        )

        coords_ax = fig.add_subplot(grid_spec[1, :-1])
        colours = [plt.get_cmap('twilight_shifted')(1.*i/255) for i in range(256)]
        ts = np.linspace(-1, 1, 256)
        positions = [0, 31, 63, 95, 127, 159, 191, 223, 255]

        gradients = np.vstack((ts, ts))
        coords_ax.imshow(gradients, aspect='auto', cmap="twilight_shifted")
        coords_ax.tick_params(labelleft=False, left=False)
        coords_ax.set_xticks(positions, labels=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

        log.info(f"Loading sample {file_path}")
        wav = data.load_sample(file_path)
        x = vae.pre_process(wav.unsqueeze(0)).squeeze()
        samples_per_second = int(np.ceil(vae.front_end.sample_rate / vae.front_end.fft_hop_length))

        bbox = [2530, 2564, 20, 30]
        log.info(f"Extracting call at bounding box: {bbox}")
        t_i, t_j, f_i, f_j = bbox
        signal = x[t_i:t_j, f_i:f_j]
        s = signal.size(0)

        log.info(f"Removing foreground")
        x_bg = designal(x)[:192]
        t_start = (x_bg.size(0)) // 2
        x_bg[t_start - s:t_start, f_i:f_j] = signal

        xs = []
        deltas = np.linspace(-1.0, 1.0, 9)
        log.info(f"Applying translations")
        for i, delta in enumerate(deltas):
            x_trans = translation(x_bg.transpose(-1, -2).unsqueeze(0).unsqueeze(0), torch.tensor(delta)).squeeze().transpose(-1, -2)
            xs.append(x_trans)

        log.info(f"Plotting translations")
        for i, (x_bg, delta) in enumerate(zip(xs, deltas)):
            ax = fig.add_subplot(grid_spec[0, i])
            twin_ax = ax.twiny()
            im = model.front_end.plot(x_bg.T, cmap="Greys", ax=ax)
            ax.set_xticks([0, 95, 191], [-1.0, 0.0, 1.0])
            twin_ax.set_xticks([0, 95, 191], [0.0, 1.536/2, 1.536])
            # ax.set_title(rf"$\delta = {{{np.format_float_positional(delta, precision=2, min_digits=2)}}}$")
            ax.set_xlabel("")
            ax.set_ylabel("")
            # ax.axvline(x=95, linestyle="dashed", color="black", linewidth=3.0)
            colour = colours[positions[i]]
            # ax.axvline(x=[0, 23, 47, 71, 95, 119, 143, 167, 191][i], linestyle="dashed", color=colour, linewidth=3.0) 
            if i != 0:
                ax.tick_params(labelleft=False, left=False)

        log.info(f"Encoding translations")
        deltas = np.linspace(-1, 1, 105)
        x = torch.cat([
            translation(x_bg.transpose(-1, -2).unsqueeze(0).unsqueeze(0), torch.tensor(delta)).transpose(-1, -2)
            for delta in deltas
        ], dim=0)
        with torch.no_grad():
            vae_q_z, *_ = vae.encode(x)
            sivae_q_z, *_ = sivae.encode(x)
        vae_mu_z, _ = vae_q_z.chunk(2, dim=-1)
        vae_mu_z = vae_mu_z.squeeze()
        sivae_mu_z, _ = sivae_q_z.chunk(2, dim=-1)
        sivae_mu_z = sivae_mu_z.squeeze()

        log.info(f"Calculating gradient of z w.r.t. translation (dz/dT)")
        vae_dzdT = central_finite_difference(vae_mu_z.t(), padding_mode="circular")
        sivae_dzdT = central_finite_difference(sivae_mu_z.t(), padding_mode="circular")

        bound = max(vae_dzdT.min().abs(), vae_dzdT.max().abs(), sivae_dzdT.min().abs(), sivae_dzdT.max().abs())
        imshow_params = dict(
            origin="lower",
            aspect="auto",
            vmin=-bound,
            vmax=bound,
            cmap=sns.color_palette("vlag", as_cmap=True)
        )

        log.info(f"Plotting derivatives")
        ax = fig.add_subplot(grid_spec[2, :-1])
        im = ax.imshow(vae_dzdT, **imshow_params)
        ax.tick_params(labelbottom=False, bottom=False)
        ax.set_yticks(np.arange(0, 128 + 16, 16), np.arange(0, 128 + 16, 16))
        for p in np.arange(0, 105, 13):
            ax.axvline(x=p + 0.5, linestyle="dashed", color="black", linewidth=1.0)
            ax.axvline(x=p - 0.5, linestyle="dashed", color="black", linewidth=1.0)
        ax.set_ylabel("VAE\n\nLatent Dimension")

        ax = fig.add_subplot(grid_spec[3, :-1])
        im = ax.imshow(sivae_dzdT, **imshow_params)
        ax.tick_params(labelbottom=False, bottom=False)
        ax.set_yticks(np.arange(0, 128 + 16, 16), np.arange(0, 128 + 16, 16))
        for p in np.arange(0, 105, 13):
            ax.axvline(x=p + 0.5, linestyle="dashed", color="black", linewidth=1.0)
            ax.axvline(x=p - 0.5, linestyle="dashed", color="black", linewidth=1.0)
        ax.set_ylabel("SIVAE\n\nLatent Dimension")
        ax.set_xlabel(r"Shift ($\delta$)")

        cbar = fig.colorbar(im, cax=fig.add_subplot(grid_spec[2:, -1]))
        cbar.set_label(r"$\frac{dz}{dT}$", rotation=0)

        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        log.info(f"Saving figure to {self.save_path}")
        fig.savefig(self.save_path)

