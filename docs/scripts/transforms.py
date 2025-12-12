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

def spectrogram_params():
    with open(rootutils.find_root() / "config" / "transforms" / "log_mel_spectrogram.yaml", "r") as f:
        return yaml.safe_load(f.read())["log_mel_spectrogram"]

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

    uk_species = ["Turdus merula_Eurasian Blackbird", "Erithacus rubecula_European Robin", "Phasianus colchicus_Ring-necked Pheasant",  "Troglodytes hiemalis_Winter Wren",  "Cyanistes caeruleus_Eurasian Blue Tit",  "Columba palumbus_Common Wood-Pigeon",  "Corvus corone_Carrion Crow"]
    ec_species = ["Coereba flaveola_Bananaquit","Poliocrania exsul_Chestnut-backed Antbird","Ramphocelus flammigerus_Flame-rumped Tanager","Leptotila pallida_Pallid Dove","Capsiempis flaveola_Yellow Tyrannulet","Mionectes oleagineus_Ochre-bellied Flycatcher","Microbates cinereiventris_Tawny-faced Gnatwren","Manacus manacus_White-bearded Manakin"]
    pr_species = ["Coereba flaveola_Bananaquit","Turdus plumbeus_Red-legged Thrush","Setophaga angelae_Elfin Woods Warbler","Vireo altiloquus_Black-whiskered Vireo","Patagioenas squamosa_Scaly-naped Pigeon","Nesospingus speculiferus_Puerto Rican Tanager","Melanerpes portoricensis_Puerto Rican Woodpecker"]
    vmax, vmin = 0.0, -80

    for model_dict, species_list in [(uk, uk_species), (ec, ec_species), (pr, pr_species)]:
        dataset = model_dict["scope"].split("_")[0]
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
        for species in species_list:
            common_name = species.split("_")[1]
            W = clf[species_name].to(device)
            norm = torch.linalg.norm(W)
            for delta in torch.arange(1, 31):
                fig, ax = plt.subplots(figsize=(3, 3))
                z_tilde = z0 + ((z0 @ W.T / norm) + delta) * (W / norm)
                x_tilde = vae.decode(z_tilde, torch.ones(1, 1, 1, device=z0.device) * delta).cpu()
                x_tilde_db = (20 * np.log10(x_tilde.exp())).squeeze().t()
                plot_mel_spectrogram(
                    x_tilde_db,
                    **spectrogram_params(),
                    vmax=vmax,
                    vmin=vmin,
                    cmap="Greys",
                    ax=ax
                )
                fig.savefig(save_dir / f"{common_name}_{delta}")

if __name__ == "__main__":
    main()
