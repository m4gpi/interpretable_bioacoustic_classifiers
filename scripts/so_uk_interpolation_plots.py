import argparse
import hydra
import numpy as np
import pandas as pd
import pathlib
import torch
import rootutils
import logging
import warnings
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

from collections import defaultdict
from matplotlib import pyplot as plt
from pathlib import Path
from torchvision import transforms as T
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.data.soundscape_embeddings import SoundscapeEmbeddingsDataModule
from src.core.models.species_detector import SpeciesDetector
from src.core.models.vae import VAE
from src.core.models.sivae import SIVAE
from src.core.data.sounding_out_chorus import SoundingOutChorus
from src.core.utils.sketch import plot_mel_spectrogram, make_ax_invisible
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.core.utils import tree
from src.cli.utils.instantiators import instantiate_transforms

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def wrap_sentence(sentence, max_length):
    words = sentence.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + (1 if current_line else 0) <= max_length:
            if current_line:
                current_line += " "
            current_line += word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)

plt.rcParams.update({
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
    'legend.fontsize': 13,
})

species_params = [
    # { "species_name": "Sylvia atricapilla_Eurasian Blackcap", "file_name": "PL-11_0_20150603_0645.wav", "t_start_seconds": 19, "delta": 30,  },
    { "species_name": "Turdus merula_Eurasian Blackbird", "file_name": "train/data/PL-12_0_20150604_0345.wav", "t_start_seconds": 4.3, "delta": 10,  },
    { "species_name": "Erithacus rubecula_European Robin", "file_name": "train/data/BA-04_0_20150620_0515.wav", "t_start_seconds": 29, "delta": 15,  },
    { "species_name": "Phasianus colchicus_Ring-necked Pheasant", "file_name": "train/data/PL-03_0_20150605_0445.wav", "t_start_seconds": 29.5, "delta": 10,  },
    { "species_name": "Troglodytes hiemalis_Winter Wren", "file_name": "train/data/PL-11_0_20150605_0500.wav", "t_start_seconds": 45.2, "delta": 15,  },
    { "species_name": "Cyanistes caeruleus_Eurasian Blue Tit", "file_name": "train/data/PL-12_0_20150604_0345.wav", "t_start_seconds": 47,"delta": 10,  },
    { "species_name": "Columba palumbus_Common Wood-Pigeon", "file_name": "train/data/BA-04_0_20150620_0615.wav", "t_start_seconds": 24.2,  "delta": 10,  },
    { "species_name": "Corvus corone_Carrion Crow", "file_name": "train/data/BA-01_0_20150621_0445.wav", "t_start_seconds": 32.6, "delta": 10,  }
]

# having these hard coded is a pain
model_map = {
    "tan-ohio": "sivae",
    "brave-vincent": "sivae",
    "small-peru": "sivae",
    "uncanny-burma": "sivae",
    "detailed-ticket": "sivae",
    "mossy-andrea": "sivae",
    "silly-byte": "vae",
    "meek-zebra": "vae",
    "rude-money": "vae",
    "misty-lecture": "vae",
    "ultimate-story": "vae",
    "tusked-chief": "vae",
}

# having these hard coded is a pain
seed_map = {
    "tan-ohio": 8,
    "small-peru": 16,
    "brave-vincent": 24,
    "uncanny-burma": 8,
    "detailed-ticket": 16,
    "mossy-andrea": 24,
    "silly-byte": 8,
    "meek-zebra": 16,
    "rude-money": 24,
    "tusked-chief": 8,
    "ultimate-story": 16,
    "misty-lecture": 24,
}

def main(
    data_dir: Path,
    scores_path: Path,
    save_dir: Path,
    seed_num: int = 0,
    device_id: int = 0,
) -> None:
    device = f"cuda:{device_id}" if device_id is not None else "cpu"
    # load final test scores
    log.info("loading scores")
    df = pd.read_parquet(scores_path).reset_index()
    df = df[df["scope"] == "SO_UK"]
    df["version"] = df["model"]
    df["model"] = df["version"].map(model_map)
    df["seed"] = df["version"].map(seed_map)
    df["model_class"] = df["model"].str.upper()
    df["model_class"] = pd.Categorical(df["model_class"], categories=["VAE", "SIVAE"], ordered=True)
    scores = df.copy()
    df = df.drop_duplicates(subset=["model", "version", "seed"])[["model", "version", "seed", "model_class"]]
    df.loc[df["model"] == "sivae", "delta"] = 0.0

    scores = scores[scores["species_name"].isin([s["species_name"] for s in species_params])]
    # init audio dataset and audio params
    hops_per_second = 48_000 / 384
    frame_length_seconds = 192 / hops_per_second
    frame_length_hops = 192
    log.info("loading SO UK dataset")
    data = SoundingOutChorus("~/data/sounding_out", sample_rate=48_000, test=False)
    # define the species we want to render examples and generate interpolations for
    log.info("identify the habitat where selected species occur most frequently")
    # identify the habitat where each species occurs most
    # for each species, we use that habitat average embedding as our background template for interpolation
    for params in species_params:
        counts = data.metadata.merge(
            data.labels.astype(bool).astype(int),
            left_index=True,
            right_index=True
        ).groupby("habitat")[params["species_name"]].sum().reset_index()
        params["habitat"] = counts.loc[counts[params["species_name"]].idxmax()].habitat
    # sort by model class for figure order, load and cache all models
    log.info("loading and caching VAEs and CLFs")
    df = df.sort_values("seed").groupby("model_class").nth(seed_num).reset_index()

    habitat_map = {"PL": "UK1", "KN": "UK2", "BA": "UK3"}
    vaes = []
    clfs = []
    z_model_habitat_means = defaultdict(tree)

    class_map = {"vae": VAE, "sivae": SIVAE}

    model_path = pathlib.Path("/its/home/kag25/models/v3")
    for i, row in df.iterrows():
        model_cls = class_map[row.model]
        vae_ckpt_path = model_path / row.model / row.version / "step=180000.ckpt"
        vae = model_cls.load_from_checkpoint(vae_ckpt_path, map_location=device)
        vaes.append(vae.eval().to(device))
        log.info(f"Loaded {row.model}/{row.version} from {vae_ckpt_path}")
        # load species logistic regression model weights
        clf = SpeciesDetector.load_from_checkpoint(model_path / "species_detectors" / f"{row.version}_SO_UK.ckpt", map_location=device).eval()
        clf = dict(zip(clf.classifiers.keys(), map(lambda layer: layer.weight, clf.classifiers.values())))
        clfs.append(clf)
        # compute the habitat model average embedding
        dm = SoundscapeEmbeddingsDataModule(root=data_dir / row.version / "SO_UK")
        dm.setup()
        # hack in the habitat labels
        embeddings = dm.train_data
        features = embeddings.features
        labels = embeddings.labels.reset_index()
        labels = labels.merge(data.metadata.loc[data.metadata.index.isin(labels.file_i), ["file_name", "habitat"]], on='file_i', how="inner")
        # encode the habitat mean representation for this model
        z_mean = embeddings.features[[f"z_mean_{i}" for i in range(128)]].merge(labels[["file_i", "habitat"]], on="file_i", how="left").drop("file_i", axis=1).groupby("habitat").mean()
        for habitat in z_mean.index:
            z_model_habitat_mean = torch.tensor(z_mean.loc[habitat], dtype=torch.float32, device=device)
            z_model_habitat_means[habitat][row.model] = z_model_habitat_mean.unsqueeze(0).unsqueeze(0)

    log.info("building plot, rendering spectrograms and interpolated reconstructions")
    # plot spectrograms and interpolated reconstructions
    vmax, vmin = 0.0, -80
    fig, axes = plt.subplots(
        nrows=len(df) + 1,
        ncols=len(species_params) + 1,
        figsize=(2 * len(species_params), 2.5 * (len(df) + 1)),
        height_ratios=[*[1.0 / (len(vaes) + 1) for _ in range(len(vaes) + 1)]],
        width_ratios=[0.01, *[0.95 / (len(species_params)) for _ in range(len(species_params))]],
        constrained_layout=True,
    )
    with torch.no_grad():
        for i, params in enumerate(species_params):
            species_name, file_name, t_start_seconds, delta, habitat = params.values()
            if i == 0:
                title_ax = axes[0, 0]
                title_ax.set_ylabel("Real\nExample")
                make_ax_invisible(title_ax)
            ax = axes[0, i + 1]
            t_start = int(t_start_seconds * hops_per_second)
            t_end = t_start + int(frame_length_seconds * hops_per_second)
            spectrogram = LogMelSpectrogram() # hard coded transform matches the specific parametrisation of the trained models
            x = spectrogram(data.load_sample(data.base_dir / file_name).unsqueeze(0)).squeeze()
            x_db = 20 * np.log10(x[:, t_start:t_end].exp())
            # plot the original
            spectrogram.plot(x_db, vmax=vmax, vmin=vmin, ax=ax, cmap="Greys")
            ax.set_xticks([0, 191], [0.0, 1.536], rotation=90)
            ax.set_title(wrap_sentence(species_name.replace("_", ", "), 10))
            ax.tick_params(labelbottom=False, bottom=False)
            ax.set_xlabel("")
            if i != 0:
                ax.tick_params(labelleft=False, left=False)
                ax.set_ylabel("")
            for j, row in df.iterrows():
                vae, clf = vaes[j], clfs[j]
                if i == 0:
                    title_ax = axes[j + 1, 0]
                    title_ax.set_ylabel(row.model_class)
                    make_ax_invisible(title_ax)
                # load scores for this species and mode;
                model_species_scores = scores[
                    (scores["model"] == row["model"]) &
                    (scores["version"] == row["version"]) &
                    (scores["species_name"] == species_name)
                ].iloc[0]
                # fetch habitat silent embedding
                z = z_model_habitat_means[habitat][row.model]
                # fetch weights of log reg model
                log.info(f"generating {species_name} with {row.model}:{row.version}")
                W = clf[species_name]
                # slightly different realisations by adding little noise to the representation and decode
                # linear interpolation across the hyperplane by delta
                # delta needs to be tuned per species
                norm = torch.linalg.norm(W)
                z_tilde = z + ((z @ W.T / norm) + delta) * (W / norm)
                # decode using VAE with alignment factor dt
                if row.model == "vae":
                    x_tilde = vae.decode(z_tilde).cpu()
                else:
                    # dt is tunable by VAE
                    x_tilde = vae.decode(z_tilde, torch.ones(1, 1, 1, device=z.device) * row.delta).cpu()
                # map to decibels
                x_tilde_db = 20 * np.log10(x_tilde.exp())
                ax = axes[j + 1, i + 1]
                # plot reconstruction
                spectrogram.plot(x_tilde_db.squeeze().t(), vmax=vmax, vmin=vmin, cmap="Greys", ax=ax)
                ax.set_xticks([0, 191], [0.0, 1.536], rotation=90)
                if j != len(df) - 1:
                    ax.tick_params(labelbottom=False, bottom=False)
                    ax.set_xlabel("")
                if i != 0:
                    ax.tick_params(labelleft=False, left=False)
                    ax.set_ylabel("")
                AP = np.format_float_positional(model_species_scores['AP'], precision=2)
                auROC = np.format_float_positional(model_species_scores['auROC'], precision=2)
                ax.set_title(f"AP: {AP}\nauROC: {auROC}")
    fig.suptitle("SO UK")
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"so_uk_{seed_num}_interpolation_w_scores.pdf", format="pdf", bbox_inches="tight")
    log.info(f"figure saved to {(save_dir / f'so_uk_{seed_num}_interpolation_w_scores.pdf').expanduser()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--scores-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p),
        required=False,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--seed-num",
        type=int,
        default=0,
        help="model number",
    )
    args = parser.parse_args()
    main(**vars(args))
