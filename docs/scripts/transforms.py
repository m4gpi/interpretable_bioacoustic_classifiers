import argparse
import hydra
import numpy as np
import pandas as pd
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
from src.core.models.base_vae import BaseVAE
from src.core.models.smooth_nifti_vae import SmoothNiftiVAE
from src.core.data.sounding_out_chorus import SoundingOutChorus
from src.core.utils.sketch import plot_mel_spectrogram, make_ax_invisible
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.core.utils import tree
from src.cli.utils.instantiators import instantiate_transforms

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

def get_spectrogram():
    with open(rootutils.find_root() / "config" / "transforms" / "log_mel_spectrogram.yaml", "r") as f:
        conf = yaml.safe_load(f.read())["log_mel_spectrogram"]
        return hydra.utils.instantiate(conf)

@torch.no_grad()
def main():
    device = "cuda:0"
    data_dir = rootutils.find_root() / "data" / "soundscape_vae_embeddings"
    save_dir = rootutils.find_root() / "docs" / "assets" / "images" / "generated"
    save_dir.mkdir(exist_ok=True, parents=True)

    index = pd.read_parquet(data_dir / "index.parquet")
    uk = index[(index["scope"] == "SO_UK") & (index["model_name"] == "nifti_vae") & (index["version"] == "v12")].iloc[0].to_dict()
    ec = index[(index["scope"] == "SO_EC") & (index["model_name"] == "nifti_vae") & (index["version"] == "v12")].iloc[0].to_dict()
    pr = index[(index["scope"] == "RFCX_bird") & (index["model_name"] == "nifti_vae") & (index["version"] == "v14")].iloc[0].to_dict()

    so = SoundingOutChorus("/srv/thetis2/kag25/data/sounding_out")
    rfcx = RainforestConnection("/srv/thetis2/kag25/data/rainforest_connection")
    spectrogram = get_spectrogram()
    hops_per_second = spectrogram.sample_rate / spectrogram.hop_length
    spectrogram_params = dict(
        hop_length=spectrogram.hop_length,
        sample_rate=spectrogram.sample_rate,
        mel_min_hertz=spectrogram.mel_min_hertz,
        mel_max_hertz=spectrogram.mel_max_hertz,
        mel_scaling_factor=spectrogram.mel_scaling_factor,
        mel_break_frequency=spectrogram.mel_break_frequency,
        vmax=0.0,
        vmin=-80,
    )

    uk_species = [
        ("Turdus merula_Eurasian Blackbird", spectrogram(so.load_sample("train/data/PL-12_0_20150604_0345.wav"))).squeeze()[int(4.3 * hops_per_second):int(4.3 * hops_per_second) + frame_length],
        ("Erithacus rubecula_European Robin", spectrogram(so.load_sample("train/data/BA-04_0_20150620_0515.wav"))).squeeze()[int(29 * hops_per_second):int(29 * hops_per_second) + frame_length],
        ("Phasianus colchicus_Ring-necked Pheasant", spectrogram(so.load_sample("train/data/PL-03_0_20150605_0445.wav"))).squeeze()[int(29.5 * hops_per_second):int(29.5 * hops_per_second) + frame_length],
        ("Troglodytes hiemalis_Winter Wren", spectrogram(so.load_sample("train/data/PL-11_0_20150605_0500.wav"))).squeeze()[int(45.2 * hops_per_second):int(45.2 * hops_per_second) + frame_length],
        ("Cyanistes caeruleus_Eurasian Blue Tit", spectrogram(so.load_sample("train/data/PL-12_0_20150604_0345.wav"))).squeeze()[int(47.0 * hops_per_second):int(47.0 * hops_per_second) + frame_length],
        ("Columba palumbus_Common Wood-Pigeon", spectrogram(so.load_sample("train/data/BA-04_0_20150620_0615.wav"))).squeeze()[int(24.2 * hops_per_second):int(24.2 * hops_per_second) + frame_length],
        ("Corvus corone_Carrion Crow", spectrogram(so.load_sample("train/data/BA-01_0_20150621_0445.wav"))).squeeze()[int(32.6 * hops_per_second):int(32.6 * hops_per_second) + frame_length],
    ]
    ec_species = [
        ("Coereba flaveola_Bananaquit", spectrogram(so.load_sample("train/data/FS-11_0_20150806_0600.wav"))).squeeze()[int(32.4 * hops_per_second):int(32.4 * hops_per_second) + frame_length],
        ("Poliocrania exsul_Chestnut-backed Antbird", spectrogram(so.load_sample("train/data/TE-03_0_20150716_0635.wav"))).squeeze()[int(25.2 * hops_per_second):int(25.2 * hops_per_second) + frame_length],
        ("Ramphocelus flammigerus_Flame-rumped Tanager", spectrogram(so.load_sample("test/data/PO-11_0_20150815_1815.wav"))).squeeze()[int(0.0 * hops_per_second):int(0.0 * hops_per_second) + frame_length],
        ("Leptotila pallida_Pallid Dove", spectrogram(so.load_sample("train/data/PO-11_0_20150818_0625.wav"))).squeeze()[int(6.5 * hops_per_second):int(6.5 * hops_per_second) + frame_length],
        ("Capsiempis flaveola_Yellow Tyrannulet", spectrogram(so.load_sample("train/data/PO-03_0_20150815_0640.wav"))).squeeze()[int(16.0 * hops_per_second):int(16.0 * hops_per_second) + frame_length],
        ("Mionectes oleagineus_Ochre-bellied Flycatcher", spectrogram(so.load_sample("train/data/FS-16_0_20150802_0645.wav"))).squeeze()[int(0.2 * hops_per_second):int(0.2 * hops_per_second) + frame_length],
        ("Microbates cinereiventris_Tawny-faced Gnatwren", spectrogram(so.load_sample("train/data/TE-06_0_20150716_0645.wav"))).squeeze()[int(33.5 * hops_per_second):int(33.5 * hops_per_second) + frame_length],
        ("Manacus manacus_White-bearded Manakin", spectrogram(so.load_sample("train/data/FS-04_0_20150802_0650.wav"))).squeeze()[int(41.2 * hops_per_second):int(41.2 * hops_per_second) + frame_length],
    ]
    pr_species = [
        ("Coereba flaveola_Bananaquit", spectrogram(rfcx.load_sample("train/0a9cdd8a5.flac"))).squeeze()[int(49.2 * hops_per_second):int(49.2 * hops_per_second) + frame_length],
        ("Turdus plumbeus_Red-legged Thrush", spectrogram(rfcx.load_sample("train/9cb1f4a34.flac"))).squeeze()[int(28.8 * hops_per_second):int(28.8 * hops_per_second) + frame_length],
        ("Setophaga angelae_Elfin Woods Warbler", spectrogram(rfcx.load_sample("train/88b5c9c1b.flac"))).squeeze()[int(40.5 * hops_per_second):int(40.5 * hops_per_second) + frame_length],
        ("Vireo altiloquus_Black-whiskered Vireo", spectrogram(rfcx.load_sample("train/c4b778e64.flac"))).squeeze()[int(28.25 * hops_per_second):int(28.25 * hops_per_second) + frame_length],
        ("Patagioenas squamosa_Scaly-naped Pigeon", spectrogram(rfcx.load_sample("train/21e2f2977.flac"))).squeeze()[int(23.0 * hops_per_second):int(23.0 * hops_per_second) + frame_length],
        ("Nesospingus speculiferus_Puerto Rican Tanager", spectrogram(rfcx.load_sample("train/1702d35a0.flac"))).squeeze()[int(29.1 * hops_per_second):int(29.1 * hops_per_second) + frame_length],
        ("Melanerpes portoricensis_Puerto Rican Woodpecker", spectrogram(rfcx.load_sample("train/745171bf2.flac"))).squeeze()[int(4.5 * hops_per_second):int(4.5 * hops_per_second) + frame_length],
    ]

    for model_dict, species_list in [(uk, uk_species), (ec, ec_species), (pr, pr_species)]:
        dataset = model_dict["scope"].split("_")[0]
        dt = 0.0 if dataset == "SO" else 1.0
        vae = get_vae(model_dict).to(device).eval()
        checkpoint = torch.load(model_dict["clf_checkpoint_path"], map_location=device)
        clf = {param.split(".")[1]: checkpoint["state_dict"][param] for param in checkpoint["state_dict"].keys() if param.startswith("classifiers") and param.endswith("weight")}
        dm = SoundscapeEmbeddingsDataModule(
            root=data_dir,
            model=model_dict["model_name"],
            version=model_dict["version"],
            scope=model_dict["scope"],
        )
        dm.setup()
        z0 = torch.tensor(dm.train_data.features.iloc[:, range(128)].mean().to_numpy(), device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        for species, spectrogram in species_list:
            common_name = species.split("_")[1]
            W = clf[species].to(device)
            norm = torch.linalg.norm(W)
            for delta in torch.arange(1, 31):
                log.info(f"{common_name}_{delta}")
                fig, ax1, ax2 = plt.subplots(figsize=(7, 4), constrained_layout=True)
                plot_mel_spectrogram((20 * np.log10(spectrogram.exp())).squeeze().t(), **spectrogram_params, cmap="Greys", ax=ax1)
                plot_mel_spectrogram
                z_tilde = z0 + ((z0 @ W.T / norm) + delta) * (W / norm)
                dt = torch.ones(1, 1, 1, device=z0.device) * dt
                x_tilde = vae.decode(z_tilde, dt).cpu()
                plot_mel_spectrogram((20 * np.log10(x_tilde.exp())).squeeze().t(), **spectrogram_params, cmap="Greys", ax=ax2)
                fig.savefig(save_dir / f"{common_name}_{delta}")

if __name__ == "__main__":
    main()
