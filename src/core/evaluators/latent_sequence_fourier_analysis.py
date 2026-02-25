import abc
import torch
import pathlib
import hydra
import itertools
import logging
import lightning as L
import numpy as np
import matplotlib as mpl
import seaborn as sns
import tqdm
import pandas as pd

from pathlib import Path
from torch.functional import F
from matplotlib import pyplot as plt
from torchvision import transforms as T
from typing import Any, List, Dict, Tuple

from src.core.utils.sketch import plot_mel_spectrogram
from src.core.evaluators.base import Evaluator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# TODO: compute fourier transform across each dimension independently, get a spectrum of the features
# stack spectrums for each dimension and plot as a heatmap

# TODO: compute histograms, fix ranges, 95% percentile of latent space, with high number of bins
# compute derivative histogram, fix ranges, 95% percentile of latent space derivatives

__all__ = ["LatentSequenceFourierAnalysis"]

class LatentSequenceFourierAnalysis(Evaluator):
    def __init__(self, results_dir: str | pathlib.Path) -> None:
        super().__init__()
        self.results_dir = pathlib.Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

    def __call__(self, trainer, model, data_module, config) -> None:
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path is not None, f"No checkpoint found at {ckpt_path}"

        log.info(f"Encoding <{config.data.get('_target_')}> with <{config.model.get('_target_')}>")
        predictions = trainer.predict(model, data_module.predict_dataloader(), ckpt_path=config.get("ckpt_path"), return_predictions=True)
        df = pd.concat(list(itertools.chain(*predictions)), axis=0)
        df = df.reset_index().merge(
            data_module.data.metadata.file_name,
            left_on="file_i",
            right_on="file_i",
            how="left"
        ).set_index(df.index.names)

        log.info(f"Computing STFT over each latent dimension")
        num_fft_hops = data_module.data.segment_len * model.sample_rate / model.fft_hop_length
        frame_length_seconds = (1 / model.sample_rate * model.fft_hop_length) * model.frame_window_length
        seq_len = int(data_module.data.segment_len / frame_length_seconds)
        mu = df.values[:, :model.latent_dim].reshape(-1, int(seq_len), model.latent_dim)
        sampling_freq = 1 / frame_length_seconds
        nyquist = sampling_freq / 2
        freqs = np.fft.rfftfreq(seq_len, frame_length_seconds)
        window = np.hanning(seq_len)
        with tqdm.tqdm(total=mu.shape[0]) as pbar:
            for x, file_name in zip(mu, df.file_name.unique()):
                fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(7, 11), width_ratios=[0.95, 0.05], constrained_layout=True)
                metadata = data_module.data.metadata[data_module.data.metadata.file_name == file_name].iloc[0]
                spectrogram = data_module.transforms(data_module.data.load_sample(metadata.file_path)).squeeze()
                mesh = plot_mel_spectrogram(spectrogram.T, **model.spectrogram_params, vmin=spectrogram.min(), vmax=spectrogram.max(), ax=axes[0, 0])
                fig.colorbar(mesh, cax=axes[0, 1], orientation="vertical")
                mag, phase = self.spectra(x, window)
                im = self.plot_spectra(mag, freqs, ax=axes[1, 0], cmap=sns.color_palette("light:b", as_cmap=True))
                cbar = fig.colorbar(im, cax=axes[1, 1], orientation="vertical")
                cbar.set_label("Magnitude")
                im = self.plot_spectra(phase, freqs, ax=axes[2, 0], cmap=plt.get_cmap("twilight"))
                cbar = fig.colorbar(im, cax=axes[2, 1], orientation="vertical")
                cbar.set_label("Phase")
                hist = torch.zeros(10, 128)
                bins = torch.linspace(-3.5, 3.5, 10 + 1)
                for j in range(10):
                    hist[j, ...] = ((bins[j] < x) & (x < bins[j + 1])).sum(axis=0) / seq_len
                    hist = torch.softmax((hist + 1e-8).log(), dim=0)
                im = axes[3, 0].imshow(
                    hist.t(),
                    extent=[-3.5, 3.5, 1, 128],
                    cmap=sns.color_palette("magma", as_cmap=True),
                    aspect="auto",
                    interpolation="none",
                    vmin=0.0,
                    vmax=1.0,
                )
                ax.tick_params(axis='x', rotation=90)
                ax.set_xticks(np.linspace(-3.5, 3.5, 10 + 1))
                ax.set_yticks(np.arange(0, 127, 4))
                cbar = fig.colorbar(im, cax=axes[3, 1], orientation="vertical")
                fig.suptitle("Spectra of Latent Timeseries Means")
                fig.savefig(self.results_dir / f"{file_name}.png")
                plt.close()
                pbar.update(1)
        log.info(f"Saved to {str(self.results_dir)}")

    @staticmethod
    def spectra(x, window):
        spectra = []
        for j in range(x.shape[-1]):
            spectrum = np.fft.rfft(x[:, j] * window)
            spectra.append(spectrum)
        spectra = np.vstack(spectra)
        mag = np.abs(spectra)
        phase = np.angle(spectra)
        return mag, phase

    @staticmethod
    def plot_spectra(Z, fq, ax, cmap):
        xx, yy = np.meshgrid(np.arange(Z.shape[0]), fq)
        mesh = ax.pcolormesh(xx, yy, Z.T, cmap=cmap)
        ax.set_xticks(np.arange(0, Z.shape[0], 4) + 1)
        ax.set_yticks(fq)
        ax.tick_params(axis='x', rotation=90)
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Frequency (Hz)")
        return mesh

def main(config: DictConfig):
    log.info("Instantiating transforms...")
    transforms: List[L.Callback] = instantiate_transforms(config.get("transforms"))

    log.info(f"Instantiating datamodule <{config.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(config.data, transforms=transforms)
    data_module.setup()

    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model, **data_module.data.model_params)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(config.trainer)

    evaluator = LatentSequenceFourierAnalysis(results_dir=pathlib.Path("./models/tssi_vae.pt:v2/latent_sequence_fourier_analysis"))
    evaluator(trainer, model, data_module, config)

if __name__ == "__main__":
    main()


