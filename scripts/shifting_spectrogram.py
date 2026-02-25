import torch
import hydra
import rootutils
import numpy as np
import yaml

from matplotlib import pyplot as plt

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.data.sounding_out_chorus import SoundingOutChorus
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.core.transforms.translation import translation
from src.core.models.tssi_vae import TSSIVAE
from src.cli.utils.instantiators import instantiate_transforms

def get_transforms():
    with open(rootutils.find_root() / "config" / "transforms" / "cropped_log_mel_spectrogram.yaml", "r") as f:
        transform_conf = yaml.safe_load(f.read())
        transforms = instantiate_transforms(transform_conf)
        log_mel_spectrogram_params = transform_conf["log_mel_spectrogram"]
        del log_mel_spectrogram_params["_target_"]
    return transforms

def get_data(transforms):
    with open(rootutils.find_root() / "config" / "data" / "sounding_out_chorus.yaml", "r") as f:
        conf = yaml.safe_load(f.read())
        data = hydra.utils.instantiate(conf, root="/mnt/data0/kag25/sounding_out", transforms=transforms)
    return data

@torch.no_grad()
def main():
    transforms = get_transforms()
    model = TSSIVAE.load_from_checkpoint("./models/tssi_vae.pt:v1/model.ckpt")
    dm = get_data(transforms)
    dm.setup()
    data = dm.test_data
    deltas = []
    for x, *_ in data:
        x = x.to(model.device).float()
        x_framed = x.view(x.size(0), -1, 1, model.frame_window_length, x.size(-1)).flatten(end_dim=1)
        xfr = x_framed[0].unsqueeze(0)
        x_shifts = torch.cat([translation(xfr, delta.view(1, 1, 1, 1)) for delta in torch.linspace(-1, 1, 192, device=xfr.device)], dim=0)
        _, delta = model.encode(x_shifts)
        delta = delta.flatten().cpu().numpy()
        deltas.append(delta)
    delta = np.concatenate(deltas)
    plt.hist(delta, bins=20, density=True)
    plt.savefig("delta_hist_dist.png")

if __name__ == "__main__":
    main()
