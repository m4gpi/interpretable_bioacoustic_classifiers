import argparse
import hydra
import numpy as np
import pandas as pd
import pathlib
import torch
import rootutils
import seaborn as sns
import logging
import warnings
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib import lines
from matplotlib import colors as mcolors
from pathlib import Path
from torchvision import transforms as T
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.data.sounding_out_chorus import SoundingOutChorus
from src.core.utils.sketch import plot_mel_spectrogram
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def get_vae(model_dict):
    with open(rootutils.find_root() / "config" / "model" / f"{model_dict['model_name']}.yaml", "r") as f:
        model_conf = yaml.safe_load(f.read())
        vae = hydra.utils.instantiate(model_conf)
    checkpoint = torch.load(model_dict["vae_checkpoint_path"], map_location="cpu")
    vae.load_state_dict(checkpoint["model_state_dict"])
    log.info(f"Loaded {model_dict['model_name']} from {model_dict['vae_checkpoint_path']}")
    return vae

def get_transforms():
    with open(rootutils.find_root() / "config" / "transforms" / "cropped_log_mel_spectrogram.yaml", "r") as f:
        transform_conf = yaml.safe_load(f.read())
        transforms = instantiate_transforms(transform_conf)
        log_mel_spectrogram_params = transform_conf["log_mel_spectrogram"]
        del log_mel_spectrogram_params["_target_"]
    return transforms

plt.rcParams.update({
    'font.size': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


def main(
    data_dir: pathlib.Path,
    audio_dir: pathlib.Path,
    save_dir: pathlib.Path,
) -> None:
    data = SoundingOutChorus(audio_dir, test=True)

    df = pd.read_parquet(data_dir / "index.parquet")
    model_dict = df[(df.model_name == "nifti_vae") & (df.version == "v12") & (df.scope == "SO_UK")].iloc[0].to_dict()
    vae = get_vae(model_dict).to(device).eval()

    transforms = get_transforms()
    spectrogram = transforms.transforms[0]
    spectrogram_params = dict(
        hop_length=spectrogram.hop_length,
        sample_rate=spectrogram.sample_rate,
        mel_min_hertz=spectrogram.mel_min_hertz,
        mel_max_hertz=spectrogram.mel_max_hertz,
        mel_scaling_factor=spectrogram.mel_scaling_factor,
        mel_break_frequency=spectrogram.mel_break_frequency,
    )

    file_names = [
        "PL-14_0_20150604_0630.wav",
        "KN-13_0_20150509_0530.wav",
        "BA-01_0_20150619_0500.wav",
        "TE-03_0_20150713_1830.wav",
        "FS-08_0_20150806_0630.wav",
        "PO-02_0_20150818_1845.wav",
    ]
    start_times = [55, 19, 9, 25, 7, 30]
    num_samples = 9
    time = np.linspace(-1, 1, num_samples)

    width = 20
    height = 2 * len(file_names)
    fig = plt.figure(figsize=(width, height), constrained_layout=True, dpi=150)
    grid_spec = fig.add_gridspec(
        nrows=1 + len(file_names),
        ncols=3 + num_samples + 1,
        width_ratios=[*[0.1 for i in range(3 + num_samples)], 0.01],
        height_ratios=[0.02, *[0.98 / 6 for i in range(len(file_names))]],
        hspace=0.2, wspace=0.05,
    )
    frame_duration_seconds = vae_frame_length * spectrogram.stft_hop_length_seconds
    window_duration_seconds = 2 * frame_duration_seconds
    hops_per_second = int(1 / spectrogram.stft_hop_length_seconds)
    window_end_hops = window_duration_seconds * hops_per_second

    # time co-ordinates
    coords_ax = fig.add_subplot(grid_spec[0, 3:-1])
    colours = [plt.get_cmap('twilight_shifted')(1.*i/255) for i in range(256)]
    ts = np.linspace(-1, 1, 256)
    positions = [0, 31, 63, 95, 127, 159, 191, 223, 255]
    coords_ax_2 = coords_ax.twiny()
    gradients = np.vstack((ts, ts))
    coords_ax.imshow(gradients, aspect='auto', cmap="twilight_shifted")
    coords_ax.tick_params(labelleft=False, left=False)
    coords_ax.set_xticks(positions, labels=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    coords_ax_2.set_xticks(positions, labels=[np.format_float_positional(x, precision=3) for x in np.arange(num_samples) * frame_duration_seconds / (num_samples - 1)])

    results = []
    for i, (file_name, frame_start_seconds) in enumerate(zip(file_names, start_times)):
        vmin, vmax = np.inf, -np.inf
        idx = df[df.file_name == file_name].index[0]
        # fetch spectrogram
        wav = data[idx].x
        log_mel = spectrogram(wav).squeeze()
        # fetch relevant bounds
        frame_end_seconds = frame_start_seconds + (1.536)
        frame_start_hops = int(frame_start_seconds * hops_per_second)
        frame_end_hops = int(frame_end_seconds * hops_per_second)
        window_start_seconds = frame_start_seconds - frame_duration_seconds / 2
        window_end_seconds = frame_end_seconds + frame_duration_seconds / 2
        window_start_hops = int(window_start_seconds * hops_per_second)
        window_end_hops = int(window_end_seconds * hops_per_second)
        # original, show the center frame with half either side
        x = log_mel[window_start_hops:window_end_hops]
        mel_original = 20 * np.log10(x.exp().numpy().T)
        vmin = min(mel_original.min(), vmin)
        vmax = max(mel_original.max(), vmax)
        # encode 3 full frames with center the one of interest
        t_start = frame_start_hops - int(frame_duration_seconds * hops_per_second)
        t_end = frame_end_hops + int(frame_duration_seconds * hops_per_second)
        q_z_window, _, dt_hat = vae.encode(log_mel[t_start:t_end].unsqueeze(0).unsqueeze(0).detach())
        x0 = vae.decode(q_z_window.chunk(2, dim=-1)[0], dt_hat).detach().squeeze()
        mel_recon = 20 * np.log10(x0.exp().numpy().T)
        # reconstruction, show the center frame with half frame either side
        mel_recon = mel_recon[:, 92:480]
        # extract the prototype encoding (center frame)
        q_z_frame = q_z_window[:, 1, :].unsqueeze(1)
        x1 = vae.decode(q_z_frame.chunk(2, dim=-1)[0]).detach().squeeze()
        # show the prototype, with reconstruction on neighbours frames (half either side)
        x = torch.cat([x0[96:192], x1, x0[384:480]], dim=0)
        mel_prototype = 20 * np.log10(x.exp().numpy().T)
        vmin = min(mel_prototype.min(), vmin)
        vmax = max(mel_prototype.max(), vmax)
        # instantiate different versions by parameterising a shift
        mel_instances = []
        for dt in time:
            x2 = vae.decode(q_z_frame.chunk(2, dim=-1)[0], torch.ones(1, 1, 1) * dt).detach().squeeze()
            x = torch.cat([x0[0:96], x2, x0[384:480]], dim=0)
            mel_instance = 20 * np.log10(x.exp().numpy().T)
            vmin = min(mel_instance.min(), vmin)
            vmax = max(mel_instance.max(), vmax)
            mel_instances.append(mel_instance)
        results.append((mel_original, mel_recon, dt_hat[:, 1, :], mel_prototype, mel_instances, (vmin, vmax)))

    for i, result in enumerate(results):
        mel_original, mel_recon, dt_hat, mel_prototype, mel_instances, (vmin, vmax) = result
        ax = fig.add_subplot(grid_spec[1 + i, 0])
        img = plot_mel_spectrogram(mel_original, **spectrogram_params, vmin=vmin, vmax=vmax, ax=ax, cmap="Greys")
        ax.axvline(x=384 // 4, color="white", linestyle="dashed", linewidth=3.0)
        ax.axvline(x=192 + 384 // 4, color="white", linestyle="dashed", linewidth=3.0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(bottom=False, labelbottom=False)

        if i == len(results) - 1:
            ax.set_xticks([0, 96, 288, 384], labels=["", 0.0, 1.536, ""])
            ax.tick_params(bottom=True, labelbottom=True)

        ax = fig.add_subplot(grid_spec[1 + i, 1])
        img = plot_mel_spectrogram(mel_recon, **spectrogram_params, vmin=vmin, vmax=vmax, ax=ax, cmap="Greys")
        ax.axvline(x=384 // 4, color="white", linestyle="dashed", linewidth=3.0)
        ax.axvline(x=192 + 384 // 4, color="white", linestyle="dashed", linewidth=3.0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        if i == 0:
            ax.set_title(rf"$\Delta \hat{{t}} = {{{np.format_float_positional(dt_hat.item(), precision=2, min_digits=2)}}}$")
            # ax.set_title("Reconstruction\n" + rf"$\Delta \hat{{t}} = {{{np.format_float_positional(dt_hat.item(), precision=2)}}}$")
        else:
            ax.set_title(rf"$\Delta \hat{{t}} = {{{np.format_float_positional(dt_hat.item(), precision=2, min_digits=2)}}}$")
        if i == len(results) - 1:
            ax.set_xticks([0, 96, 288, 384], labels=["", 0.0, 1.536, ""])
            ax.tick_params(bottom=True, labelbottom=True)

        ax = fig.add_subplot(grid_spec[1 + i, 2])
        img = plot_mel_spectrogram(mel_prototype, **spectrogram_params, vmin=vmin, vmax=vmax, ax=ax, cmap="Greys")
        ax.axvline(x=384 // 4, color="white", linestyle="dashed", linewidth=3.0)
        ax.axvline(x=192 + 384 // 4, color="white", linestyle="dashed", linewidth=3.0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        # if i == 0:
        #     ax.set_title("Prototype\n")
        if i == len(results) - 1:
            ax.set_xticks([0, 96, 288, 384], labels=["", 0.0, 1.536, ""])
            ax.tick_params(bottom=True, labelbottom=True)

        for j, (mel_instance, dt) in enumerate(zip(mel_instances, time)):
            ax = fig.add_subplot(grid_spec[1 + i, 3 + j])
            img = plot_mel_spectrogram(mel_instance, **spectrogram_params, vmin=vmin, vmax=vmax, ax=ax, cmap="Greys")
            colour = colours[positions[j]]
            ax.axvline(x=96, color=colour, linestyle="dashed", linewidth=3.0)
            ax.axvline(x=192 + 96, color=colour, linestyle="dashed", linewidth=3.0)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
            if i == 0:
                ax.set_title(rf"$\Delta \hat{{t}} = {{{np.format_float_positional(dt, precision=2, min_digits=2)}}}$")
            if i == len(results) - 1:
                ax.set_xticks([0, 96, 288, 384], labels=["", 0.0, 1.536, ""])
                ax.tick_params(bottom=True, labelbottom=True)
        print(i, len(results) - 1)
        cbar_ax = fig.add_subplot(grid_spec[1 + i, -1])
        cbar = fig.colorbar(img, cax=cbar_ax, orientation="vertical", format="%+3.1f dB")

    save_file = save_dir / f"shifted_calls.pdf"
    print(save_file)
    fig.savefig(save_file, format="pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/embeddings/dir",
    )
    parser.add_argument(
        "--audio-dir",
        type=lambda p: Path(p),
        required=False,
        help="/path/to/audio/dir/",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p),
        required=False,
        help="/path/to/saved/",
    )
    args = parser.parse_args()
    main(**vars(args))
