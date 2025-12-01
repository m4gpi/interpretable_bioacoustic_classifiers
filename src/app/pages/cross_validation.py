import attrs
import dash
import dash_mantine_components as dmc
import logging
import pandas as pd
import pathlib
import pyarrow.dataset as ds
import pyarrow as pa
import rootutils

from dash import Dash, dcc, ctx, html
from dash import Output, Input, State, callback, no_update
from omegaconf import DictConfig
from plotly import express as px
from plotly import graph_objects as go

from typing import List

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["App"]

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

@attrs.define()
class App:
    results_dir: pathlib.Path = attrs.field(converter=lambda p: pathlib.Path(p).expanduser())

    results_df: pd.DataFrame = attrs.field(init=False)
    summary_df: pd.DataFrame = attrs.field(init=False)

    def setup(self, cfg: DictConfig) -> dash.Dash:
        self.prepare_data()
        log.info("Building...")
        app = dash.Dash(__name__)
        app.layout = self.layout()
        self.register_callbacks(app)
        return app

    def prepare_data(self):
        df = ds.dataset(
            self.results_dir / "val_scores.parquet",
            format="parquet",
            schema=pa.schema({
                "model": pa.string(),
                "version": pa.string(),
                "scope": pa.string(),
                "pool_method": pa.string(),
                "run_id": pa.string(),
                "fold_id": pa.int64(),
                "clf_learning_rate": pa.float64(),
                "l1_penalty": pa.float64(),
                "attn_learning_rate": pa.float64(),
                "attn_weight_decay": pa.float64(),
                "key_per_target": pa.bool8(),
                "species_name": pa.string(),
                "AP": pa.float64(),
                "auROC": pa.float64(),
                "epoch": pa.int64(),
            }),
        ).to_table().to_pandas()
        # drop runs without key_per_target
        df = (
            df[((df["pool_method"] == "prob_attn") & (df["key_per_target"] == True)) | (df.pool_method.isin(["mean", "max"]))]
            .drop_duplicates(
                subset=[
                    "model", "version", "scope", "pool_method",
                    "clf_learning_rate", "l1_penalty", "attn_learning_rate", "attn_weight_decay",
                    "key_per_target", "fold_id",
                ],
                keep="first"
            )
        )
        # first, across all model types, pick the epoch with the best model-class cross-fold score
        kfolds_df = (
            df.groupby([
                "model", "version", "scope", "pool_method",
                "clf_learning_rate", "l1_penalty",
                "attn_learning_rate", "attn_weight_decay",
                "epoch"
            ], dropna=False)
            [["auROC", "AP"]]
            .mean()
            .reset_index()
        )
        model_df = (
            kfolds_df.groupby([
                "model", "scope", "pool_method",
                "clf_learning_rate", "l1_penalty",
                "attn_learning_rate", "attn_weight_decay",
                "epoch"
            ], dropna=False)
            [["auROC", "AP"]]
            .mean()
            .reset_index()
        )
        # sum the mAUROC and mAP as a simple heuristic to pick the best
        model_df["score"] = model_df["auROC"] + model_df["AP"]
        best_epoch_idx = (
            model_df.groupby([
                "model", "scope", "pool_method",
                "clf_learning_rate", "l1_penalty",
                "attn_learning_rate", "attn_weight_decay"
            ], dropna=False)
            ["score"]
            .idxmax()
        )
        best_epoch_df = model_df.iloc[best_epoch_idx]
        df = df.merge(
            best_epoch_df[["model", "scope", "pool_method", "clf_learning_rate", "attn_learning_rate", "l1_penalty", "attn_weight_decay", "epoch"]],
            on=["model", "scope", "pool_method", "clf_learning_rate", "attn_learning_rate", "l1_penalty", "attn_weight_decay", "epoch"],
            how="inner"
        )
        # second, for prob_attn models select across clf_learning_rate and l1_penalty those that have the best model-class cross-fold score
        # we should end up with the same number of attention results as max and mean
        attn_df = df[df["pool_method"] == "prob_attn"]
        attn_kfolds_df = (
            attn_df.groupby([
                "model", "version", "scope", "pool_method",
                "clf_learning_rate", "l1_penalty",
                "attn_learning_rate", "attn_weight_decay",
                "epoch"
            ], dropna=False)
            [["auROC", "AP"]]
            .mean()
            .reset_index()
        )
        attn_model_df = (
            attn_kfolds_df.groupby([
                "model", "scope", "pool_method",
                "clf_learning_rate", "l1_penalty",
                "attn_learning_rate", "attn_weight_decay",
                "epoch"
            ], dropna=False)
            [["auROC", "AP"]]
            .mean()
            .reset_index()
        )
        # sum the mAUROC and mAP as a simple heuristic to pick the best
        attn_model_df["score"] = attn_model_df["auROC"] + attn_model_df["AP"]
        best_attn_params_idx = (
            attn_model_df.groupby([
                "model", "scope", "pool_method",
                "clf_learning_rate", "l1_penalty"
            ])
            ["score"]
            .idxmax()
        )
        best_attn_params_df = attn_model_df.iloc[best_attn_params_idx]
        # recombine results with max and mean
        results_df = pd.concat([
            attn_df.merge(
                best_attn_params_df[["model", "scope", "pool_method", "clf_learning_rate", "attn_learning_rate", "l1_penalty", "attn_weight_decay", "epoch"]],
                on=["model", "scope", "pool_method", "clf_learning_rate", "attn_learning_rate", "l1_penalty", "attn_weight_decay",  "epoch"],
                how="inner"
            ),
            df[df.pool_method.isin(["max", "mean"])],
        ], axis=0)
        # factorise by shared hyperparameters, ignoring epoch and attention parameters because we've already sub-selected
        column, groups = pd.factorize(results_df[["model", "scope", "clf_learning_rate", "l1_penalty"]].apply(tuple, axis=1))
        results_df["parameter_group"] = column.astype(str)
        # build a summary table
        summary_df = (
            results_df.groupby([
                "model", "scope", "pool_method",
                "clf_learning_rate", "l1_penalty",
                "attn_learning_rate", "attn_weight_decay",
                "epoch", "parameter_group"
            ], dropna=False)
            [["auROC", "AP"]]
            .mean()
            .reset_index()
        )
        summary_df["score"] = summary_df["AP"] + summary_df["auROC"]
        summary_df = summary_df.sort_values(by="score", ascending=False)
        summary_df["auROC"] = summary_df["auROC"].round(4)
        summary_df["AP"] = summary_df["AP"].round(4)
        summary_df["score"] = summary_df["score"].round(4)

        self.results_df = results_df
        self.summary_df = summary_df

    def layout(self):
        return dmc.MantineProvider(
            children=dmc.AppShell([
                dmc.Title("Cross Validation Scores", ta="center"),
                dmc.Group(
                    grow=True,
                    children=[
                        dmc.Box([
                            dmc.Text("Select a dataset"),
                            dmc.Select(
                                id="dataset-select",
                                data=self.results_df["scope"].unique(),
                            ),
                        ]),
                        dmc.Box([
                            dmc.Text("Select a model"),
                            dmc.Select(
                                id="model-select",
                            ),
                        ]),
                        dmc.Box([
                            dmc.Chip(
                                id="show-points",
                                children="Show Points"
                            ),
                        ]),
                        dmc.Button(
                            id="render-button",
                            children="Render",
                        ),
                    ]
                ),
                dmc.Group(
                    grow=True,
                    children=[
                        dcc.Loading(
                            dmc.Box(id="summary-table"),
                        ),
                    ],
                ),
                dmc.Group(
                    grow=True,
                    children=[
                        dcc.Loading(
                            dcc.Graph(id="average-precision-graph", figure=empty_figure())
                        ),
                    ]
                ),
                dmc.Group(
                    grow=True,
                    children=[
                        dcc.Loading(
                            dcc.Graph(id="roc-auc-graph", figure=empty_figure())
                        ),
                    ]
                ),
            ])
        )

    def register_callbacks(self, app):
        @app.callback(
            Output("model-select", "data"),
            Input("dataset-select", "value"),
        )
        def set_model_select(scope: str) -> List[str]:
            return self.results_df.loc[self.results_df["scope"] == scope, "model"].unique()

        @app.callback(
            Output("summary-table", "children"),
            State("dataset-select", "value"),
            State("model-select", "value"),
            Input("render-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def show_summary(
            scope: str,
            model: str,
            n_clicks: int,
        ) -> dmc.Table:
            data = self.summary_df[(self.summary_df["scope"] == scope) & (self.summary_df["model"] == model)].iloc[:10]
            caption = "Best model parameters using 5-folds cross validation across 3 random seeds"
            return dmc.Table(
                layout="fixed",
                style={"width": "100%"},
                children=[
                    dmc.TableThead(
                        dmc.TableTr([
                            dmc.TableTh(col)
                            for col in data.columns
                        ])
                    ),
                    dmc.TableTbody([
                        dmc.TableTr([
                            dmc.TableTd(record[col])
                            for col in data.columns
                        ])
                        for record in data.to_dict(orient="records")
                    ]),
                    dmc.TableCaption(caption)
                ]
            )

        @app.callback(
            Output("average-precision-graph", "figure"),
            State("dataset-select", "value"),
            State("model-select", "value"),
            Input("render-button", "n_clicks"),
            Input("show-points", "checked"),
            prevent_initial_call=True,
        )
        def plot_average_precision_scores(
            scope: str,
            model: str,
            n_clicks: int,
            show_points: bool,
        ) -> go.Figure:
            log.info("plotting AP")
            params = dict(
                x="parameter_group",
                y="AP",
                color="pool_method",
                hover_data=[
                    "species_name",
                    "AP",
                    "auROC",
                    "fold_id",
                    "clf_learning_rate",
                    "l1_penalty",
                    "attn_learning_rate",
                    "attn_weight_decay",
                    "epoch",
                    "parameter_group",
                ],
                category_orders=dict(
                    parameter_group=self.summary_df.parameter_group.tolist(),
                ),
            )
            if show_points:
                fig = px.strip(self.results_df, **params)
            else:
                fig = px.violin(self.results_df, **params, box=True)
                fig.update_traces(meanline_visible=True)
            fig.update_layout(height=400, title_text="Average Precision (AP)")
            return fig

        @app.callback(
            Output("roc-auc-graph", "figure"),
            State("dataset-select", "value"),
            State("model-select", "value"),
            Input("render-button", "n_clicks"),
            Input("show-points", "checked"),
            prevent_initial_call=True,
        )
        def plot_roc_auc_scores(
            model: str,
            scope: str,
            n_clicks: int,
            show_points: bool,
        ) -> go.Figure:
            log.info("plotting ROC AUC")
            params = dict(
                x="parameter_group",
                y="auROC",
                color="pool_method",
                hover_data=[
                    "species_name",
                    "AP",
                    "auROC",
                    "fold_id",
                    "clf_learning_rate",
                    "l1_penalty",
                    "attn_learning_rate",
                    "attn_weight_decay",
                    "epoch",
                    "parameter_group",
                ],
                category_orders=dict(
                    parameter_group=self.summary_df.parameter_group.tolist(),
                ),
            )
            if show_points:
                fig = px.strip(self.results_df, **params)
            else:
                fig = px.violin(self.results_df, **params, box=True)
                fig.update_traces(meanline_visible=True)
            fig.update_layout(height=400, title_text="ROC AUC")
            return fig


