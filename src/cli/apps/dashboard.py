import attrs
import base64
import cmasher as cmr
import dash
import dash_mantine_components as dmc
import hydra
import importlib
import librosa
import lightning as L
import logging
import numpy as np
import pandas as pd
import pathlib
import plotly.express as px
import plotly.graph_objs as go
import rootutils
import sklearn
import soundfile as sf
import torch
import tqdm
import warnings

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

from dash import Dash, dcc, ctx, html
from dash import Output, Input, State, callback, no_update
from matplotlib import colors as mcolors
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from src.core.utils.log_mel_spectrogram import LogMelSpectrogram
from src.core.utils import metrics
from src.cli.utils.instantiators import instantiate_transforms

__all__ = ["App"]

cmap = cmr.lavender
plotly_colorscale = [[i/255, mcolors.to_hex(cmap(i/255))] for i in range(256)]

def empty_figure():
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="gray")
    )
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
    )
    return fig

def encode_audio(wav, sr):
    import io
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format='wav')
    buffer.seek(0)
    b64_wav = base64.b64encode(buffer.read()).decode('ascii')
    return f"data:audio/wav;base64,{b64_wav}"

@torch.no_grad()
def encode(
    model: L.LightningModule,
    data_loader: L.LightningDataModule,
    dataset: torch.utils.data.Dataset,
    stage: str,
) -> Tuple[NDArray, NDArray, NDArray]:
    file_i, labels, predictions, attn_w = [], [], [], []
    for batch in tqdm.tqdm(data_loader):
        x, y, s, y_freq = batch
        _, y_probs, A, _, _ = model.model_step((x.to(model.device), y.to(model.device), s, y_freq))
        file_i.append(s)
        labels.append(y)
        predictions.append(y_probs)
        attn_w.append(A)
    file_i, labels, predictions, attn_w = [torch.cat(x, dim=-1).cpu().numpy() for x in [file_i, labels, predictions, attn_w]]
    predictions, attn_w  = predictions.mean(axis=1), attn_w.mean(axis=1)
    # interleave attention weights to get correct timestep weighting
    seq_len = attn_w.shape[1] // 2
    W = np.empty(attn_w.shape)
    # when we roll forward, the last element goes to the beginning,
    # so we need to put that first
    W[:, 1::2, :] = attn_w[:, :seq_len, :]
    W[:, 0::2, :] = attn_w[:, seq_len:, :]
    attn_w = W
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
    df = pd.DataFrame(data, columns=multi_cols, index=dataset.labels.loc[file_i].index)
    index_columns = df.index.names
    df = df.reset_index()
    df["stage"] = stage
    df = df.set_index([*index_columns, "stage"])
    scores = []
    for label_name in label_names:
        label_df = df[label_name]
        y = label_df["label"].to_numpy()
        y_prob = label_df["prediction"].to_numpy()
        if np.isnan(y_prob).any():
            prop_nans = np.isnan(y_prob).sum() / len(y_prob)
            log.warning(f"NaNs found in predicted probabilities for {label_name} with a proportional count of {prop_nans}")
            y_prob = np.nan_to_num(y_prob, nan=0.0)
        assert not np.isnan(y).any(), f"NaNs found in true labels for {label_name}"
        scores.append(dict(
            label_name=label_name,
            AP=metrics.average_precision(y, y_prob),
            auROC=sklearn.metrics.roc_auc_score(y, y_prob),
            label_frequency=dataset.y_freq[label_name].item(),
        ))
    scores = pd.DataFrame(scores)
    scores["stage"] = stage
    return df, scores

@attrs.define()
class App:
    audio_dir: pathlib.Path = attrs.field(converter=lambda p: pathlib.Path(p).expanduser(), validator=lambda *_, p: p.exists())
    ckpt_path: pathlib.Path = attrs.field(converter=lambda p: pathlib.Path(p).expanduser(), validator=lambda *_, p: p.exists())
    model_class: str = attrs.field(default=None, validator=attrs.validators.instance_of(str))

    def setup(self, cfg: DictConfig) -> dash.Dash:
        log.info(f"Instantiating data <{cfg.data._target_}>")
        data_module = hydra.utils.instantiate(cfg.data)
        data_module.setup(stage="eval")

        log.info(f"Importing model <{self.model_class}>")
        module, class_name = ".".join(self.model_class.split(".")[:-1]), self.model_class.split(".")[-1]
        ModelClass = getattr(importlib.import_module(module), class_name)

        log.info(f"Loading checkpoint <{self.model_checkpoint_path}>")
        # HACK:
        checkpoint = torch.load(self.model_checkpoint_path)
        del checkpoint["hyper_parameters"]["species_list_path"]
        model = ModelClass(**{**checkpoint["hyper_parameters"], **data_module.data.model_params})
        model.load_state_dict(checkpoint["state_dict"])
        # model = ModelClass.load_from_checkpoint(self.model_checkpoint_path)

        transforms = instantiate_transforms(cfg.transforms)
        log_mel_spectrogram = transforms["spectrogram"]

        # hard coded VAE audio resolution parameters
        duration = 59.904
        seq_len = 78
        frame_length = 192
        frame_length_seconds = 1 / (log_mel_spectrogram.sample_rate // log_mel_spectrogram.hop_length) * frame_length

        species_names = np.array(list(data_module.data.y_freq.keys()))[list(reversed(np.array(list(data_module.data.y_freq.values())).argsort()))]
        log.info(f"Encoding training subset of <{cfg.data._target_}> with <{self.model_class}>")
        train_df, train_scores_df = encode(model, data_module.train_dataloader(), data_module.data, "train")
        log.info(f"Encoding test subset of <{cfg.data._target_}> with <{self.model_class}>")
        test_df, test_scores_df = encode(model, data_module.test_dataloader(), data_module.test_data, "test")
        data = pd.concat([train_df, test_df], axis=0).sort_index(level=1)
        log.info(f"Extracting scores")
        scores = pd.concat([train_scores_df, test_scores_df], axis=0)

        log.info(f"Launching Dash application")
        app = dash.Dash(__name__)
        app.layout = dmc.MantineProvider(
            children=dmc.AppShell(
                id="app",
                p="md",
                children=[
                    dcc.Store("current-sample", data={}),
                    dcc.Store("bounding-box", data={}),
                    dmc.Grid([
                        dmc.GridCol(
                            span=4,
                            children=[
                                dmc.Text("Select a species:", ta="left"),
                                dmc.Select(
                                    id="species-select",
                                    value=species_names[0],
                                    data=[
                                        {"label": species_name.replace("_", ", "), "value": species_name}
                                        for species_name in species_names
                                    ],
                                    searchable=True,
                                    allowDeselect=False,
                                    clearable=False,
                                ),
                            ]
                        ),
                        dmc.GridCol(
                            span=4,
                            children=[
                                dmc.Group([
                                    dmc.Stack([
                                        dmc.Box([
                                            dmc.Text("Train AP: ", span=True),
                                            dmc.Text(id="train-average-precision-score", span=True),
                                        ]),
                                        dmc.Box([
                                            dmc.Text("Test AP: ", span=True),
                                            dmc.Text(id="test-average-precision-score", span=True),
                                        ]),
                                    ]),
                                    dmc.Stack([
                                        dmc.Box([
                                            dmc.Text("Train ROC AUC: ", span=True),
                                            dmc.Text(id="train-roc-auc-score", span=True),
                                        ]),
                                        dmc.Box([
                                            dmc.Text("Test ROC AUC: ", span=True),
                                            dmc.Text(id="test-roc-auc-score", span=True),
                                        ]),
                                    ]),
                                    dmc.Stack([
                                        dmc.Box([
                                            dmc.Text("Train Label Count: ", span=True),
                                            dmc.Text(id="train-label-frequency", span=True),
                                        ]),
                                        dmc.Box([
                                            dmc.Text("Test Label Count: ", span=True),
                                            dmc.Text(id="test-label-frequency", span=True),
                                        ]),
                                    ]),
                                ]),
                            ],
                        ),
                        dmc.GridCol(
                            span=3,
                            children=[
                                dmc.Text("Select a file:", ta="left"),
                                dmc.Select(
                                    id="file-select",
                                    value=None,
                                    data=[
                                        {"label": file_name, "value": str(file_i)}
                                        for file_i, file_name in zip(data.index.get_level_values(0), data.index.get_level_values(1))
                                    ],
                                    searchable=True,
                                    allowDeselect=False,
                                    clearable=False,
                                ),
                            ]
                        ),
                        dmc.GridCol(
                            span=1,
                            children=[
                                dmc.Button(
                                    id="reset-button",
                                    children="Reset",
                                )
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
                                            figure=empty_figure(),
                                        ),
                                    ]),
                                ]),
                            ]
                        ),
                        dmc.GridCol(
                            span=6,
                            children=[
                                dmc.Stack([
                                    dmc.Box(
                                        id="sample-data",
                                        children=[],
                                    ),
                                    dcc.Loading([
                                        dmc.Group(
                                            grow=True,
                                            children=dmc.Stack([
                                                dmc.Text("Timestep Attention Weights", ta="center", size="xl"),
                                                dcc.Graph(
                                                    id="attention-weights-graph",
                                                    figure=empty_figure(),
                                                    style=dict(height=150),
                                                )
                                            ]),
                                        ),
                                    ]),
                                    dmc.Space(h="xs"),
                                    dcc.Loading([
                                        dmc.Group(
                                            grow=True,
                                            children=dmc.Stack([
                                                dmc.Text("Audio", ta="center", size="xl"),
                                                html.Audio(
                                                    id="audio-player",
                                                    src=None,
                                                controls=True,
                                                ),
                                            ]),
                                        ),
                                    ]),
                                    dmc.Space(h="xs"),
                                    dcc.Loading([
                                        dmc.Group(
                                            grow=True,
                                            children=dmc.Stack([
                                                dmc.Text("Mel Spectrogram", ta="center", size="xl"),
                                                dcc.Graph(
                                                    id="spectrogram-graph",
                                                    figure=empty_figure(),
                                                    style=dict(height=300),
                                                )
                                            ]),
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
            Output("train-average-precision-score", "children"),
            Output("train-roc-auc-score", "children"),
            Output("train-label-frequency", "children"),
            Output("test-average-precision-score", "children"),
            Output("test-roc-auc-score", "children"),
            Output("test-label-frequency", "children"),
            Input("species-select", "value"),
        )
        def set_scores(species_name):
            train_scores = scores.loc[(scores.label_name == species_name) & (scores.stage == "train")].iloc[0]
            test_scores = scores.loc[(scores.label_name == species_name) & (scores.stage == "test")].iloc[0]
            return (
                train_scores["AP"].round(3), train_scores["auROC"].round(3), train_scores["label_frequency"],
                test_scores["AP"].round(3), test_scores["auROC"].round(3), test_scores["label_frequency"],
            )

        @callback(
            Output("scatter-graph", "figure"),
            Input("species-select", "value"),
        )
        def draw_figure(species_name):
            df = data[species_name].copy()
            labels, weights = df["label"], df[[f"weight_{i}" for i in range(39 * 2)]]
            # FIXME: note this is not invariant to audio length, should be data driven
            # we might have sequences with fewer weights due to a shorter duration for other datasets
            timesteps = np.linspace(-(frame_length_seconds / 2), duration - frame_length_seconds, seq_len)
            # positive samples
            centroid = np.dot(weights[labels == 1], timesteps)
            dispersion = np.sqrt(np.sum(weights[labels == 1] * (timesteps - centroid[:, None])**2, axis=1))
            df.loc[labels == 1, "centroid"] = centroid
            df.loc[labels == 1, "dispersion"] = dispersion
            # negative samples
            centroid = np.dot(weights[labels == 0], timesteps)
            dispersion = np.sqrt(np.sum(weights[labels == 0] * (timesteps - centroid[:, None])**2, axis=1))
            df.loc[labels == 0, "centroid"] = centroid
            df.loc[labels == 0, "dispersion"] = dispersion
            # force labels to categorical
            df["label"] = pd.Categorical(df["label"], categories=[0, 1], ordered=True)
            df["label_name"] = df["label"].map({0: "Absent", 1: "Present"})
            df["prediction"] = df["prediction"].round(3)
            # TODO: map centroid and dispersion to seconds with circular boundary
            df = df.reset_index()
            fig = px.scatter(
                df,
                x="centroid",
                y="dispersion",
                symbol="label_name",
                color="prediction",
                facet_row="stage",
                opacity=0.75,
                hover_name="file_i",
                hover_data=["file_i", "file_name", "label", "prediction"],
                symbol_map={"Absent": "circle", "Present": "x"},
                category_orders=dict(label_name=["Absent", "Present"], stage=["train", "test"]),
                color_continuous_scale=plotly_colorscale,
                labels=dict(prediction="p(y)", label_name="Label")
            )
            fig.update_layout(
                height=700,
                width=800,
                margin=dict(t=0, l=80, r=200, b=0, pad=0),
                xaxis_title_text="Timestep Centroid (seconds)",
                yaxis_title_text="Timestep Dispersion (seconds)",
                yaxis2_title_text="Timestep Dispersion (seconds)",
                legend_y=1.1,
                legend_orientation='h',
                coloraxis_colorbar_title_side="right",
            )
            return fig

        @callback(
            Output("bounding-box", "data", allow_duplicate=True),
            Input("reset-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def reset_sample(n_clicks):
            return {}

        @callback(
            Output("file-select", "value"),
            Input("scatter-graph", "clickData"),
        )
        def store_current_file(clicked_data):
            if clicked_data is None or len((points := clicked_data["points"])) == 0:
                return no_update
            return str(points[0]["hovertext"])

        @callback(
            Output("current-sample", "data"),
            Output("bounding-box", "data"),
            Input("file-select", "value"),
            Input("species-select", "value"),
            State("current-sample", "data"),
            State("bounding-box", "data"),
            prevent_initial_call=True,
        )
        def store_current_sample(file_i, species_name, current_sample, bounding_box):
            if not file_i or not species_name:
                return no_update
            sample = data.loc[int(file_i), species_name].reset_index().iloc[0].to_dict()
            if len(current_sample) and current_sample == sample:
                return no_update, bounding_box
            return sample, {}

        @callback(
            Output("sample-data", "children"),
            Input("current-sample", "data"),
        )
        def set_file_name(sample):
            if not sample:
                return no_update
            return dmc.Group(
                grow=True,
                children=[
                    dmc.Box([
                        dmc.Text(f"Label: {sample['label']}", ta="center"),
                    ]),
                    dmc.Box([
                        dmc.Text(f"Predicted Probability: {round(sample['prediction'], 3)}", ta="center"),
                    ]),
                ]
            )

        @callback(
            Output("audio-player", "src"),
            Input("current-sample", "data"),
            Input("bounding-box", "data"),
        )
        def set_audio_content(sample, bounding_box):
            if not sample:
                return no_update
            file_path = self.audio_dir / sample["stage"] / "data" / sample["file_name"]
            wav, sr = librosa.load(file_path, sr=log_mel_spectrogram.sample_rate, duration=duration)
            wav = np.concatenate((wav[-(frame_length // 2) * log_mel_spectrogram.hop_length:], wav))
            if bounding_box:
                start_idx = int((bounding_box["x0"] + frame_length_seconds / 2) * sr)
                num_samples = int(frame_length_seconds * sr)
            else:
                start_idx = 0
                num_samples = int((duration + frame_length_seconds / 2) * sr)
            return encode_audio(wav[start_idx:start_idx + num_samples], sr)

        @callback(
            Output("spectrogram-graph", "figure"),
            Input("current-sample", "data"),
            Input("bounding-box", "data"),
        )
        def draw_spectrogram(sample, bounding_box):
            if not sample:
                return no_update
            file_path = self.audio_dir / sample["stage"] / "data" / sample["file_name"]
            wav, sr = librosa.load(file_path, sr=log_mel_spectrogram.sample_rate, duration=duration)
            log_mel = log_mel_spectrogram(wav)
            # circular pad the beginning with the overlap from the end
            log_mel = np.concatenate((log_mel[:, -(frame_length // 2):], log_mel), axis=1)
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                x=np.linspace(-(frame_length_seconds / 2), duration, log_mel.shape[1]),
                y=np.arange(0, 64),
                z=20 * np.log10(np.exp(log_mel)),
                colorscale='viridis',
                colorbar=dict(tickformat="%+3.1f dB", title_text="Magnitude (dB)", title_side="right"),
            ))
            if bounding_box:
                t_start, t_end = bounding_box["x0"], bounding_box["x1"]
                zoom_region = [t_start - frame_length_seconds, t_end + frame_length_seconds]
                fig["layout"]["shapes"] = [bounding_box]
                fig["layout"]["xaxis"]["range"] = zoom_region
                fig["layout"]["xaxis"]["autorange"] = False
            fig.update_layout(
                height=250,
                xaxis_title_text="Time (seconds)",
                yaxis_title_text="Mel Bin",
                margin=dict(t=0, l=0, r=0, b=0, pad=0),
            )
            return fig

        @callback(
            Output("attention-weights-graph", "figure"),
            Input("current-sample", "data"),
        )
        def draw_attention_weights(sample):
            if not sample:
                return no_update
            fig = go.Figure()
            w = np.array([sample[f"weight_{i}"] for i in range(seq_len)])
            x = np.linspace(-(frame_length_seconds / 2), duration - frame_length_seconds, seq_len)
            trace = go.Bar(x=x, y=w)
            fig.add_trace(trace)
            fig.update_layout(
                height=150,
                xaxis_title_text="Time (seconds)",
                yaxis_range=[0.0, 1.0],
                margin=dict(t=0, l=0, r=0, b=0, pad=0),
            )
            return fig

        @callback(
            Output("bounding-box", "data", allow_duplicate=True),
            Input("attention-weights-graph", "clickData"),
            prevent_initial_call=True,
        )
        def set_bounding_box(clicked_data):
            log.info(clicked_data)
            if clicked_data is None or len((points := clicked_data["points"])) == 0:
                return no_update
            t_start = points[0]["x"]
            t_end = t_start + frame_length_seconds
            return dict(
                type="rect",
                x0=t_start,
                x1=t_end,
                y0=0, y1=63,
                line=dict(color="black", width=3),
                fillcolor="rgba(0,0,0,0)",
            )
        return app
