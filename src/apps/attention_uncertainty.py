import argparse
import base64
import dash
import dash_mantine_components as dmc
import librosa
import lightning as L
import logging
import os
import numpy as np
import pandas as pd
import pathlib
import plotly.express as px
import plotly.graph_objs as go
import rootutils
import soundfile as sf
import torch
import tqdm

from dash import Dash, dcc, ctx, html
from dash import Output, Input, State, callback, no_update
from numpy.typing import NDArray
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from src.core.data.soundscape_vae_embeddings import SoundscapeVAEEmbeddingsDataModule
from src.core.models.species_detector import SpeciesDetector
from src.core.utils.log_mel_spectrogram import LogMelSpectrogram

def encode_audio(wav, sr):
    import io
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format='wav')
    buffer.seek(0)
    b64_wav = base64.b64encode(buffer.read()).decode('ascii')
    return f"data:audio/wav;base64,{b64_wav}"

def encode(
    model: L.LightningModule,
    data_loader: L.LightningDataModule,
    dataset: torch.utils.data.Dataset,
    stage: str,
) -> Tuple[NDArray, NDArray, NDArray]:
    labels = []
    file_indices = []
    predictions = []
    attn_w = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            x, y, s, y_freq = batch
            x, y = x.to(model.device), y.to(model.device)
            species_names = list(y_freq.keys())
            label_frequency = list(y_freq.values())
            A_V = torch.tanh(model.attention_V(x))
            A_U = torch.sigmoid(model.attention_U(x))
            for species_name in species_names:
                clf = model.classifiers[species_name]
                attention_w = model.attention_w[species_name]
                A = F.softmax(attention_w(A_V * A_U), dim=-2) # (N, T, 1)
                y_species_logits = clf((x * A).sum(dim=-2))
                y_species_probs = torch.sigmoid(y_species_logits)
                predictions.append(y_species_probs)
                attn_w.append(A)
            labels.append(y)
            file_indices.append(s)
    predictions = torch.cat(predictions, dim=-1).mean(dim=1).cpu().numpy()
    attn_w = torch.cat(attn_w, dim=-1).mean(dim=1).cpu().numpy()
    labels = torch.cat(labels, dim=-1).cpu().numpy()
    file_indices = torch.cat(file_indices, dim=-1).cpu().numpy()
    label_names = dataset.labels.columns
    data_blocks = []
    columns = []
    for i, name in enumerate(label_names):
        block = np.column_stack([
            labels[:, i],
            predictions[:, i],
            attn_w[:, :, i]
        ])
        data_blocks.append(block)
        cols = [(name, "label"), (name, "prediction")] + [(name, f"weight_{j}") for j in range(attn_w.shape[1])]
        columns.extend(cols)
    data = np.concatenate(data_blocks, axis=1)
    multi_cols = pd.MultiIndex.from_tuples(columns, names=["label_name", "feature_type"])
    df = pd.DataFrame(data, columns=multi_cols, index=dataset.labels.loc[file_indices].index)
    index_columns = df.index.names
    df = df.reset_index()
    df["stage"] = stage
    return df.set_index([*index_columns, "stage"])

def create_dash_app(
    root: pathlib.Path,
    model: str,
    version: str,
    scope: str,
    audio_dir: pathlib.Path,
    ckpt_path: pathlib.Path,
    device: int,
) -> dash.Dash:
    data_module = SoundscapeVAEEmbeddingsDataModule(root=root, model=model, version=version, scope=scope).setup(stage="eval")
    model = SpeciesDetector.load_from_checkpoint(ckpt_path)
    log_mel_spectrogram = LogMelSpectrogram()
    species_names = np.array(list(data_module.data.y_freq.keys()))[list(reversed(np.array(list(data_module.data.y_freq.values())).argsort()))]
    train_df = encode(model, data_module.train_dataloader(), data_module.data, "train")
    test_df = encode(model, data_module.test_dataloader(), data_module.test_data, "test")
    data = pd.concat([train_df, test_df], axis=0)

    app = dash.Dash(__name__)
    app.layout = dmc.MantineProvider(
        children=dmc.AppShell(
            id="app",
            p="md",
            children=[
                dcc.Store("current-sample", data={}),
                dcc.Store("audio-bounds", data=[59.904, 0.0]),
                dmc.Grid([
                    dmc.GridCol(
                        span=12,
                        children=[
                            dmc.Text("Select a species:", ta="left"),
                            dmc.Select(
                                id="species-select",
                                value=species_names[0],
                                data=species_names,
                                allowDeselect=False,
                                clearable=False,
                            ),
                        ]
                    )
                ]),
                dmc.Grid([
                    dmc.GridCol(
                        span=6,
                        children=[
                            dcc.Loading([
                                dmc.Stack([
                                    dmc.Box([
                                        dmc.Text("Timestep Attention Centroid & Dispersion", ta="center", size="xl"),
                                    ]),
                                    dcc.Graph(
                                        id="scatter-graph",
                                        figure=None,
                                    ),
                                ]),
                            ]),
                        ]
                    ),
                    dmc.GridCol(
                        span=6,
                        children=[
                            dmc.Stack([
                                dmc.Box([
                                    dmc.Text(id="file-name", ta="center", size="xl"),
                                ]),
                                dcc.Loading([
                                    dmc.Center([
                                        html.Audio(
                                            id="audio-player",
                                            src=None,
                                            controls=True,
                                        ),
                                    ]),
                                ]),
                                dcc.Loading([
                                    dmc.Box(
                                        id="attention-weights-container",
                                        children=dcc.Graph(
                                            id="attention-weights-graph",
                                            figure=None,
                                            style=dict(height=200),
                                        )
                                    ),
                                ]),
                                dcc.Loading([
                                    dmc.Box(
                                        id="spectrogram-container",
                                        children=dcc.Graph(
                                            id="spectrogram-graph",
                                            figure=None,
                                            style=dict(height=400),
                                        )
                                    ),
                                ]),
                            ]),
                        ]
                    )
                ]),
            ]
        )
    )

    @callback(
        Output("scatter-graph", "figure"),
        Input("species-select", "value"),
    )
    def draw_figure(species_name):
        T = 78
        df = data[species_name].copy()
        labels, weights = df["label"], df[[f"weight_{i}" for i in range(T)]]
        timesteps = np.arange(T)
        # positive samples
        mean = np.dot(weights[labels == 1], timesteps)
        std = np.sqrt(np.sum(weights[labels == 1] * (timesteps - mean[:, None])**2, axis=1))
        df.loc[labels == 1, "mean"] = mean
        df.loc[labels == 1, "std"] = std
        # negative samples
        mean = np.dot(weights[labels == 0], timesteps)
        std = np.sqrt(np.sum(weights[labels == 0] * (timesteps - mean[:, None])**2, axis=1))
        df.loc[labels == 0, "mean"] = mean
        df.loc[labels == 0, "std"] = std
        df["label"] = df["label"].astype(str)
        df["prediction"] = df["prediction"].round(3)
        df = df.reset_index()
        fig = px.scatter(
            df,
            x="mean",
            y="std",
            color="label",
            facet_row="stage",
            opacity=0.75,
            hover_name="file_i",
            hover_data=["file_i", "file_name", "label", "prediction"],
        )
        fig.update_layout(
            height=700,
            width=800,
            xaxis_title_text="Timestep (Centroid)",
            yaxis_title_text="Timestep (Dispersion)",
            yaxis2_title_text="Timestep (Dispersion)",
            legend_y=1.1,
            legend_orientation='h',
        )
        return fig

    @callback(
        Output("current-sample", "data"),
        Input("scatter-graph", "clickData"),
        Input("species-select", "value"),
    )
    def store_current_sample(clicked_data, species_name):
        if clicked_data is None or len((points := clicked_data["points"])) == 0:
            return no_update
        file_i = points[0]["hovertext"]
        sample = data.loc[file_i, species_name].reset_index().iloc[0]
        return sample.to_dict()

    @callback(
        Output("file-name", "children"),
        Input("current-sample", "data"),
    )
    def set_file_name(sample):
        if not sample:
            return no_update
        return sample["file_name"]

    @callback(
        Output("audio-player", "src"),
        Input("current-sample", "data"),
        Input("audio-bounds", "data"),
    )
    def set_audio_content(sample, audio_bounds):
        if not sample:
            return no_update
        file_path = audio_dir / sample["stage"] / "data" / sample["file_name"]
        duration, offset = audio_bounds
        wav, sr = librosa.load(file_path, duration=duration, offset=offset)
        return encode_audio(wav, sr)

    @callback(
        Output("spectrogram-graph", "figure"),
        Input("current-sample", "data"),
    )
    def draw_spectrogram(sample):
        if not sample:
            return no_update
        file_path = audio_dir / sample["stage"] / "data" / sample["file_name"]
        wav, sr = librosa.load(file_path, duration=59.904)
        log_mel = log_mel_spectrogram(wav)
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=np.linspace(0, 59.904, log_mel.shape[1]),
            y=np.arange(0, 64),
            z=20 * np.log10(np.exp(log_mel)),
            colorscale='viridis',
            colorbar=dict(tickformat="%+3.1f dB", title_text="Magnitude (dB)", title_side="right"),
        ))
        fig.update_layout(
            height=400,
            title_text="Log Mel Spectrogram",
            xaxis_title_text="Time (s)",
            yaxis_title_text="Mel Bin",
        )
        return fig

    @callback(
        Output("attention-weights-graph", "figure"),
        Input("current-sample", "data"),
    )
    def draw_attention_weights(sample):
        if not sample:
            return no_update
        seq_len = 78
        w = np.array([sample[f"weight_{i}"] for i in range(seq_len)])
        w = w.reshape(2, seq_len // 2).repeat(2, 1)
        w[1] = np.roll(w[1], 1)
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=np.arange(0, 59.904, 1.536 / 2),
            y=np.arange(2),
            z=w,
            zmin=0.0,
            zmax=1.0,
            colorscale='RdBu_r',
        ))
        fig.update_layout(
            height=300,
            title_text="Timestep Attention Weights",
        )
        return fig

    @callback(
        Output("spectrogram-graph", "figure", allow_duplicate=True),
        Output("audio-bounds", "data"),
        Input("attention-weights-graph", "clickData"),
        Input("spectrogram-graph", "figure"),
        prevent_initial_call=True,
    )
    def draw_bounding_box_on_spectrogram(clicked_data, fig):
        log.info(clicked_data)
        if clicked_data is None or len((points := clicked_data["points"])) == 0:
            return no_update
        t_start = points[0]["x"]
        t_end = t_start + 1.536
        bounding_box = dict(
            type="rect",
            x0=t_start, x1=t_end,
            y0=0, y1=63,
            line=dict(color="red", width=1),
            fillcolor="rgba(0,0,0,0)",
        )
        duration = t_end - t_start
        offset = t_start - 1.536
        zoom_region = [t_start - 1.536, t_end + 1.536]
        fig["layout"]["shapes"] = [bounding_box]
        fig["layout"]["xaxis"]["range"] = zoom_region
        fig["layout"]["xaxis"]["autorange"] = False
        return fig, [duration, offset]

    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="/path/to/dataset/root", type=lambda p: pathlib.Path(p).expanduser())
    parser.add_argument("--scope", help="Dataset scope")
    parser.add_argument("--model", help="VAE model type")
    parser.add_argument("--version", help="VAE model version number")
    parser.add_argument("--audio-dir", help="/path/to/audio/dir", type=lambda p: pathlib.Path(p).expanduser())
    parser.add_argument("--ckpt-path", help="model checkpoint path", type=lambda p: pathlib.Path(p).expanduser())
    parser.add_argument("--device", help="GPU device ID or CPU", type=str)
    args = parser.parse_args()
    app = create_dash_app(**vars(args))
    app.run(debug=True)
