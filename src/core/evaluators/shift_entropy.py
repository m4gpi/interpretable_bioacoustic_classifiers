import lightning as L
import logging
import pandas as pd
import pathlib
import torch
import tqdm

from omegaconf import DictConfig
from typing import Any, Callable

from src.core.transforms.frame import frame
from src.core.transforms.translation import translation_1d as translation
from src.core.evaluators.base import Evaluator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["ShiftEntropy"]

class ShiftEntropy(Evaluator):
    def __init__(
        self,
        save_path: str | pathlib.Path,
    ) -> None:
        self.save_path = pathlib.Path(save_path)

    def __call__(self, trainer: None, model: Callable, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        log.info(f"Calculating frame shift entropy for <{data_module.__class__.__name__}> with <{model.__class__.__name__}>")

        results = []
        for dataloader_idx, dataloader in enumerate(data_module.predict_dataloader()):
            with tqdm.tqdm(total=len(dataloader)) as pbar:
                pbar.set_description(f"Dataloader {dataloader_idx}")
                for batch_idx, batch in enumerate(dataloader):
                    batch = batch.to("cuda")
                    batch = model.on_after_batch_transfer(batch, dataloader_idx)
                    x = frame(batch.x, window_length=model.frame_window_length, hop_length=model.frame_window_length // 2)
                    thetas = torch.linspace(-torch.pi, torch.pi, 100, device=x.device)
                    x_flat = x.flatten(end_dim=1)
                    x_trans = torch.stack([translation(x_flat.transpose(-1, -2), (theta  / torch.pi).expand(x_flat.size(0))).transpose(-1, -2) for theta in thetas], dim=1)
                    x_trans = x_trans.unflatten(0, (x.size(0), x.size(1)))
                    frame_mse = (x.unsqueeze(2) - x_trans).pow(2).flatten(start_dim=-3).mean(dim=-1)
                    # normalise errors as probabilities, i.e. probability the shift correctly aligns the image
                    p = frame_mse / frame_mse.sum(dim=-1, keepdim=True)
                    # calculate the entropy
                    h = -((p * p.log2()).sum(dim=-1))
                    # cache results
                    bs, seq, *_ = x.size()
                    sample_idx = batch.s.unsqueeze(0).repeat(seq, 1).t()
                    seq_idx = torch.arange(seq).repeat(bs, 1)
                    dl_idx = torch.tensor(dataloader_idx).expand(bs).unsqueeze(0).repeat(seq, 1).t()
                    ref_column_types = dict(file_i=int, dataloader_idx=int, timestep=int)
                    feat_column_types = dict(h=float)
                    column_types = (ref_column_types | feat_column_types)
                    data = dict(zip(column_types.keys(), [sample_idx.flatten().cpu(), dl_idx.flatten().cpu(), seq_idx.flatten().cpu(), *h.flatten(end_dim=1).cpu().t()]))
                    df = pd.DataFrame(data=data, columns=column_types.keys()).astype(dtype=column_types).set_index(list(ref_column_types.keys()))
                    results.append(df)
                    pbar.update(1)
        df = pd.concat(results, axis=0)
        df.to_parquet(self.save_path)



