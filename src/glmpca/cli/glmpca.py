from __future__ import annotations

import logging
import sys
import warnings
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import anndata as ad
import igraph
import matplotlib as mpl
import numpy as np
import pandas as pd
import typer
from matplotlib.figure import Figure

from glmpca.ExponentialFamily import _n_workers
from glmpca.fast_poisson import DEFAULT_PENALTY, FastPoissonPCA
from glmpca.GLMPCA import DEFAULT_CHUNK_ROWS, GLMPCA

DESCRIPTION = (
    "Reduce the dimensionality of a cell-by-feature matrix with GLM-PCA.\n\n"
    "``glmpca`` fits a generalized PCA, with an exponential family distribution such "
    "as Poisson, Bernoulli, etc., to the matrix in ``.X`` of an h5ad file, with cells "
    "in rows and features in columns. The result is an updated h5ad object with:\n"
    '* ``obsm["X_glmPCA"]``: the coordinates of the cells.\n'
    '* ``varm["glmPCA_loadings"]``: the loadings of the features.\n'
    '* ``var["glmPCA_intercept"]``: the intercept of the features.\n'
    '* ``obs["glmPCA_depth"]``: the depth factor of the cells, absent with '
    "``--noDepthFactor``.\n"
    '* ``uns["glmPCA"]``: the parameters of the fit.\n\n'
    "If ``--outFileUMAP`` is given, ``glmpca`` also computes a 2D projection (UMAP) "
    "and Leiden clusters of the cells from the reduction. It stores them in "
    '``obsm["X_umap"]`` and ``obs["leiden"]``, and writes a plot file and a .tsv '
    "file with the UMAP coordinates and the cluster of each cell."
)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help=DESCRIPTION,
    context_settings={"help_option_names": []},
)

_IO = "Input / Output options"
_GLMPCA = "glmPCA options"
_OPTIMISATION = "Optimisation options"
_CLUSTERING = "Clustering options"
_PLOT = "Plot options"
_OTHER = "Other options"

AVAILABLE_PROCESSORS = _n_workers(-1)
CM_PER_INCH = 2.54


class Init(str, Enum):
    spectral = "spectral"
    random = "random"
    lsi = "lsi"


class Optimizer(str, Enum):
    adagrad = "adagrad"
    adam = "adam"
    cg = "cg"


class FamilyChoice(str, Enum):
    """Every family of GLMPCA, and the direct Poisson fit of fast_poisson.

    A test keeps this list equal to GLMFamily plus fast_poisson.
    """

    gaussian = "gaussian"
    poisson = "poisson"
    bernoulli = "bernoulli"
    negative_binomial = "negative_binomial"
    beta = "beta"
    gamma = "gamma"
    lognormal = "lognormal"
    sigmoid_beta = "sigmoid_beta"
    fast_poisson = "fast_poisson"


class PlotFileFormat(str, Enum):
    png = "png"
    jpg = "jpg"
    svg = "svg"
    pdf = "pdf"


def normalize_processors(value: str | int) -> int:
    if value == "max/2":
        return max(AVAILABLE_PROCESSORS // 2, 1)
    if value == "max":
        return AVAILABLE_PROCESSORS

    try:
        number_of_processors = int(value)
    except ValueError as exc:
        msg = f"{value} is not a valid number of processors"
        raise typer.BadParameter(msg) from exc

    if number_of_processors < 1:
        msg = f"{value} is not a valid number of processors"
        raise typer.BadParameter(msg)
    return min(number_of_processors, AVAILABLE_PROCESSORS)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"glmpca {version('glmpca')}")
        raise typer.Exit()


def help_callback(ctx: typer.Context, value: bool) -> None:
    if value:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def log_parameters(**parameters: object) -> None:
    for name, value in parameters.items():
        logging.info("%s: %s", name, value)


def configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def fail(message: str) -> typer.Exit:
    sys.stderr.write(message + "\n")
    return typer.Exit(code=1)


def umap_leiden(
    coordinates: np.ndarray, n_neighbors: int, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    import umap  # noqa: PLC0415

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, spread=5)
    embedding = reducer.fit_transform(coordinates)

    graph = reducer.graph_.tocoo()
    upper = graph.row < graph.col
    network = igraph.Graph(
        n=graph.shape[0],
        edges=np.column_stack((graph.row[upper], graph.col[upper])).tolist(),
        edge_attrs={"weight": graph.data[upper].tolist()},
    )
    partition = network.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
        n_iterations=-1,
    )
    return np.asarray(embedding), np.asarray(partition.membership)


def plot_umap(
    embedding: np.ndarray,
    clusters: np.ndarray,
    path: str,
    width: float,
    height: float,
    file_format: PlotFileFormat,
    dpi: int,
) -> None:
    figure = Figure(figsize=(width / CM_PER_INCH, height / CM_PER_INCH))
    axes = figure.subplots()
    colormap = mpl.colormaps["tab20"]
    axes.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colormap(clusters % colormap.N),
        s=min(120_000 / len(clusters), 100),
        linewidths=0,
    )
    for cluster in np.unique(clusters):
        x, y = np.median(embedding[clusters == cluster], axis=0)
        axes.text(x, y, str(cluster), ha="center", va="center", fontweight="bold")
    axes.set_title("Leiden clusters")
    axes.set_xlabel("UMAP1")
    axes.set_ylabel("UMAP2")
    axes.set_xticks([])
    axes.set_yticks([])
    figure.savefig(path, dpi=dpi, format=file_format.value)


def run_glmpca(
    model: GLMPCA, adata: ad.AnnData
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Fits the saturated-parameter model and returns the arrays to store."""
    try:
        model.fit(adata)
    except ValueError as exc:
        raise fail(str(exc)) from exc

    loadings, intercept = model.saturated_loadings_, model.saturated_intercept_
    assert loadings is not None
    assert intercept is not None
    depth = model.saturated_depth_
    return (
        model.transform(adata).numpy(),
        loadings.numpy(),
        intercept.numpy(),
        None if depth is None else depth.numpy(),
    )


def run_fast_poisson(
    model: FastPoissonPCA, adata: ad.AnnData
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Fits the counts directly by Alternating Poisson Regression."""
    logging.info(
        "fast_poisson fits the counts directly, so the optimisation options "
        "(--learningRate, --batchSize, --gamma, --nInit, --init) do not apply. Its "
        "model holds a size factor for every cell, which is what --noDepthFactor "
        "turns off for the other families."
    )
    if not model.accelerate:
        logging.info("DAAREM acceleration is off, as --noDaarem was given.")
    try:
        model.fit(adata)
    except ValueError as exc:
        raise fail(str(exc)) from exc

    scores, loadings, intercept = model.scores_, model.loadings_, model.intercept_
    assert scores is not None
    assert loadings is not None
    assert intercept is not None
    assert model.size_factors_ is not None
    logging.info(
        "fast_poisson converged in %d passes, log-likelihood %.1f.",
        len(model.log_likelihoods_),
        model.log_likelihoods_[-1],
    )
    # The size factor of a cell is the same quantity as the depth factor of the other
    # families, so it goes to the same column.
    return (
        scores.numpy(),
        loadings.numpy(),
        intercept.numpy(),
        model.size_factors_.numpy(),
    )


@app.callback(invoke_without_command=True)
def main(
    # Input / Output options
    input: Annotated[
        str,
        typer.Option(
            "-i",
            "--input",
            metavar=".h5ad",
            rich_help_panel=_IO,
            help=(
                "Input file in .h5ad format, with cells in rows and features in "
                "columns."
            ),
        ),
    ],
    out_file: Annotated[
        str,
        typer.Option(
            "-o",
            "--outFile",
            metavar=".h5ad",
            rich_help_panel=_IO,
            help="The output .h5ad file: the input with the GLM-PCA results added.",
        ),
    ],
    out_file_umap: Annotated[
        str | None,
        typer.Option(
            "-op",
            "--outFileUMAP",
            rich_help_panel=_IO,
            help=(
                "The output plot file (for UMAP). If specified, the UMAP coordinates "
                'and Leiden clusters are added to the .h5ad file as ``obsm["X_umap"]`` '
                'and ``obs["leiden"]``, and a 4-column .tsv file with the same prefix '
                "is also created with the cell IDs, raw UMAP coordinates (UMAP1 and "
                "UMAP2) and Leiden cluster number."
            ),
        ),
    ] = None,
    # glmPCA options
    n_prin_comps: Annotated[
        int,
        typer.Option(
            "-n",
            "--nPrinComps",
            rich_help_panel=_GLMPCA,
            help=(
                "Number of principal components to reduce the dimensionality to. Use "
                "a higher number for samples with more expected heterogeneity."
            ),
        ),
    ] = 20,
    glmpca_family: Annotated[
        FamilyChoice,
        typer.Option(
            "-gf",
            "--glmPCAfamily",
            metavar="FAMILY",
            rich_help_panel=_GLMPCA,
            help=(
                "The choice of exponential family distribution.\n\n"
                "One of: [bold yellow]gaussian[/bold yellow], "
                "[bold yellow]poisson[/bold yellow], "
                "[bold yellow]bernoulli[/bold yellow], "
                "[bold yellow]beta[/bold yellow], "
                "[bold yellow]gamma[/bold yellow], "
                "[bold yellow]lognormal[/bold yellow], "
                "[bold yellow]sigmoid_beta[/bold yellow], "
                "[bold yellow]negative_binomial[/bold yellow], "
                "[bold yellow]fast_poisson[/bold yellow].\n\n"
                "[bold yellow]fast_poisson[/bold yellow] fits the counts directly by "
                "Alternating Poisson Regression (Weine et al. 2024) instead of the "
                "saturated parameters, and ignores the optimisation options."
            ),
        ),
    ] = FamilyChoice.poisson,
    # Optimisation options
    max_iter: Annotated[
        int,
        typer.Option(
            "--maxIter",
            rich_help_panel=_OPTIMISATION,
            help="Maximum number of epochs.",
        ),
    ] = 100,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learningRate",
            rich_help_panel=_OPTIMISATION,
            help=(
                "Initial learning rate. If the optimisation gives NaN values, it "
                "restarts with a smaller learning rate."
            ),
        ),
    ] = 0.2,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batchSize",
            rich_help_panel=_OPTIMISATION,
            help=(
                "Number of cells in each mini-batch. If there are fewer cells, the "
                "number of cells is used."
            ),
        ),
    ] = 256,
    gamma: Annotated[
        float,
        typer.Option(
            "--gamma",
            rich_help_panel=_OPTIMISATION,
            help=(
                "Factor that multiplies the learning rate when the cost of an epoch "
                "stops falling."
            ),
        ),
    ] = 0.5,
    n_init: Annotated[
        int,
        typer.Option(
            "--nInit",
            rich_help_panel=_OPTIMISATION,
            help="Number of optimisation runs. The run with the lowest cost is kept.",
        ),
    ] = 1,
    init: Annotated[
        Init,
        typer.Option(
            "--init",
            metavar="INIT",
            rich_help_panel=_OPTIMISATION,
            help=(
                "Start values of the loadings. [bold yellow]spectral[/bold yellow] "
                "uses the SVD of the saturated parameters of a random subset of "
                "cells. [bold yellow]lsi[/bold yellow] uses the SVD of the TF-IDF "
                "of the counts of that subset, the start that LSI uses for sparse "
                "single-cell data, and it needs counts. "
                "[bold yellow]random[/bold yellow] uses a random point on the "
                "Stiefel manifold.\n\n"
                "One of: [bold yellow]spectral[/bold yellow], "
                "[bold yellow]lsi[/bold yellow], [bold yellow]random[/bold yellow]."
            ),
        ),
    ] = Init.spectral,
    optimizer: Annotated[
        Optimizer,
        typer.Option(
            "--optimizer",
            metavar="NAME",
            rich_help_panel=_OPTIMISATION,
            help=(
                "The optimiser on the Stiefel manifold.\n\n"
                "One of: [bold yellow]adagrad[/bold yellow], "
                "[bold yellow]adam[/bold yellow], [bold yellow]cg[/bold yellow] "
                "(conjugate gradients). The learning rate default is calibrated for "
                "[bold yellow]adagrad[/bold yellow], and "
                "[bold yellow]adam[/bold yellow] usually needs a smaller one.\n\n"
                "[bold yellow]cg[/bold yellow] works on the whole matrix and picks "
                "every step by a line search, so --learningRate, --gamma and "
                "--batchSize do not reach it. It stops when no step lowers the cost."
            ),
        ),
    ] = Optimizer.adagrad,
    penalty: Annotated[
        float,
        typer.Option(
            "--penalty",
            rich_help_panel=_GLMPCA,
            help=(
                "Weight of the L2 penalty on the components of ``-gf fast_poisson``. "
                "Without it the fit can diverge on sparse counts: components chase "
                "patterns of zeros towards a rate of 0, the likelihood keeps rising, "
                "and the embedding collapses onto a few cells. 0 turns it off. It "
                "does not apply to the other families."
            ),
        ),
    ] = DEFAULT_PENALTY,
    no_accelerate: Annotated[
        bool,
        typer.Option(
            "--noDaarem",
            rich_help_panel=_GLMPCA,
            help=(
                "Turn off the DAAREM acceleration of ``-gf fast_poisson``, which runs "
                "the plain algorithm of the paper. The acceleration reaches a given "
                "log-likelihood in fewer passes, at the cost of one extra pass over an "
                "n by p matrix and a history of 2 x 5 x (n + p) x (nPrinComps + 1) "
                "numbers. It does not apply to the other families."
            ),
        ),
    ] = False,
    no_depth_factor: Annotated[
        bool,
        typer.Option(
            "--noDepthFactor",
            rich_help_panel=_GLMPCA,
            help=(
                "Do not fit an offset for every cell. With the offset, which is on by "
                "default, the model holds a term for the depth of a cell beside the "
                "term for every feature, as ``-gf fast_poisson`` does, so a component "
                "does not have to carry the depth. It has its own learning rate, 1% of "
                "``--learningRate``, the same as the per-feature intercept."
            ),
        ),
    ] = False,
    tfidf: Annotated[
        bool,
        typer.Option(
            "--tfidf",
            rich_help_panel=_GLMPCA,
            help=(
                "Fit the model to the TF-IDF of the counts rather than to the counts. "
                "This is the weighting that ``--init lsi`` uses for its start, applied "
                "to the matrix itself. It needs counts, and the families whose support "
                "is bounded (bernoulli, beta, sigmoid_beta) refuse it."
            ),
        ),
    ] = False,
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunkSize",
            rich_help_panel=_OPTIMISATION,
            help=(
                "Number of cells handled at a time when the saturated parameters are "
                "computed and when a run is scored. It bounds the memory of those two "
                "steps and does not change the result. Lower it for a large dataset on "
                "a small machine."
            ),
        ),
    ] = DEFAULT_CHUNK_ROWS,
    # Clustering options
    n_neighbors: Annotated[
        int,
        typer.Option(
            "-nk",
            "--nNeighbors",
            rich_help_panel=_CLUSTERING,
            help=(
                "Number of nearest neighbours to consider for clustering and UMAP. "
                "Choose this considering the total number of cells and the expected "
                "number of clusters; smaller numbers lead to more fragmented "
                "clusters. Only used with ``--outFileUMAP``."
            ),
        ),
    ] = 30,
    cluster_resolution: Annotated[
        float,
        typer.Option(
            "-cr",
            "--clusterResolution",
            rich_help_panel=_CLUSTERING,
            help=(
                "Resolution parameter for Leiden clustering. Values lower than 1.0 "
                "result in fewer clusters, while higher values lead to splitting of "
                "clusters. In most cases the optimum is between 0.8 and 1.2. Only "
                "used with ``--outFileUMAP``."
            ),
        ),
    ] = 1.0,
    # Plot options
    plot_width: Annotated[
        float,
        typer.Option(
            "--plotWidth",
            metavar="FLOAT",
            rich_help_panel=_PLOT,
            help="Output plot width (in cm).",
        ),
    ] = 25.0,
    plot_height: Annotated[
        float,
        typer.Option(
            "--plotHeight",
            metavar="FLOAT",
            rich_help_panel=_PLOT,
            help="Output plot height (in cm).",
        ),
    ] = 25.0,
    plot_file_format: Annotated[
        PlotFileFormat,
        typer.Option(
            "--plotFileFormat",
            metavar="FORMAT",
            rich_help_panel=_PLOT,
            help=(
                "Image format type of the plot file.\n\n"
                "One of: [bold yellow]png[/bold yellow], "
                "[bold yellow]jpg[/bold yellow], "
                "[bold yellow]svg[/bold yellow], "
                "[bold yellow]pdf[/bold yellow]."
            ),
        ),
    ] = PlotFileFormat.png,
    dpi: Annotated[
        int,
        typer.Option(
            "--dpi",
            metavar="INT",
            rich_help_panel=_PLOT,
            help="Resolution of the output image, in dots per inch.",
        ),
    ] = 300,
    # Other options
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            metavar="DEVICE",
            show_default="cuda if available, else cpu",
            rich_help_panel=_OTHER,
            help=(
                "The device to fit on, for example [bold yellow]cpu[/bold yellow], "
                "[bold yellow]cuda[/bold yellow] or [bold yellow]cuda:1[/bold yellow]."
            ),
        ),
    ] = None,
    number_of_processors: Annotated[
        int,
        typer.Option(
            "-p",
            "--numberOfProcessors",
            parser=normalize_processors,
            metavar="INT",
            show_default="max",
            rich_help_panel=_OTHER,
            help=(
                "Number of processors for the per-feature fits of the "
                "[bold yellow]beta[/bold yellow], "
                "[bold yellow]sigmoid_beta[/bold yellow] and "
                '[bold yellow]gamma[/bold yellow] families. You can also type "max/2" '
                'to use half the maximum number of processors or "max" to use all '
                "available processors."
            ),
        ),
    ] = AVAILABLE_PROCESSORS,
    verbose: Annotated[
        bool,
        typer.Option(
            "-v",
            "--verbose",
            rich_help_panel=_OTHER,
            help="Set to see processing messages.",
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "-V",
            "--version",
            is_eager=True,
            expose_value=False,
            callback=version_callback,
            rich_help_panel=_OTHER,
            help="Print the program version and exit.",
        ),
    ] = False,
    help: Annotated[
        bool,
        typer.Option(
            "-h",
            "--help",
            is_eager=True,
            expose_value=False,
            callback=help_callback,
            rich_help_panel=_OTHER,
            help="Show this message and exit.",
        ),
    ] = False,
) -> int:
    if verbose:
        log_parameters(
            input=input,
            out_file=out_file,
            n_prin_comps=n_prin_comps,
            glmpca_family=glmpca_family,
            max_iter=max_iter,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            n_init=n_init,
            init=init,
            optimizer=optimizer,
            chunk_size=chunk_size,
            no_depth_factor=no_depth_factor,
            tfidf=tfidf,
            no_accelerate=no_accelerate,
            penalty=penalty,
            out_file_umap=out_file_umap,
            n_neighbors=n_neighbors,
            cluster_resolution=cluster_resolution,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_file_format=plot_file_format,
            dpi=dpi,
            device=device,
            number_of_processors=number_of_processors,
        )
    else:
        warnings.filterwarnings("ignore")

    adata = ad.read_h5ad(input)
    if adata.X is None:
        msg = f"'{input}' has no matrix in .X."
        raise fail(msg)

    if glmpca_family is FamilyChoice.fast_poisson:
        scores, loadings, intercept, depth = run_fast_poisson(
            FastPoissonPCA(
                n_pc=n_prin_comps,
                max_iter=max_iter,
                device=device,
                accelerate=not no_accelerate,
                penalty=penalty,
            ),
            adata,
        )
    else:
        scores, loadings, intercept, depth = run_glmpca(
            GLMPCA(
                n_pc=n_prin_comps,
                family=glmpca_family.value,
                max_iter=max_iter,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                n_init=n_init,
                init=init.value,
                optimizer=optimizer.value,
                n_jobs=number_of_processors,
                device=device,
                chunk_size=chunk_size,
                tfidf=tfidf,
                depth_factor=not no_depth_factor,
            ),
            adata,
        )
    adata.obsm["X_glmPCA"] = scores
    adata.varm["glmPCA_loadings"] = loadings
    adata.var["glmPCA_intercept"] = intercept
    if depth is not None:
        adata.obs["glmPCA_depth"] = depth
    adata.uns["glmPCA"] = {
        "params": {
            "n_pc": n_prin_comps,
            "family": glmpca_family.value,
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gamma": gamma,
            "n_init": n_init,
            "init": init.value,
            "optimizer": optimizer.value,
            "tfidf": tfidf,
            "depth_factor": not no_depth_factor,
            "accelerate": not no_accelerate,
            "penalty": penalty,
        },
    }

    if out_file_umap is not None:
        embedding, clusters = umap_leiden(scores, n_neighbors, cluster_resolution)
        leiden = pd.Categorical(
            clusters.astype(str),
            categories=[str(cluster) for cluster in range(clusters.max() + 1)],
        )
        adata.obsm["X_umap"] = embedding
        adata.obs["leiden"] = leiden

        plot_umap(
            embedding,
            clusters,
            out_file_umap,
            plot_width,
            plot_height,
            plot_file_format,
            dpi,
        )
        table = pd.DataFrame(
            embedding,
            index=adata.obs_names,
            columns=pd.Index(["UMAP1", "UMAP2"]),
        )
        table["cluster"] = leiden
        table.to_csv(
            Path(out_file_umap).with_suffix(".tsv"), sep="\t", index_label="Cell_ID"
        )

    adata.write_h5ad(out_file)

    return 0


def cli() -> None:
    configure_logging()
    app()


if __name__ == "__main__":
    cli()
