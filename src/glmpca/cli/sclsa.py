from __future__ import annotations

import logging
import warnings
from enum import Enum
from importlib.metadata import version
from typing import Annotated

import anndata as ad
import typer

from glmpca.cli.glmpca import (
    PlotFileFormat,
    configure_logging,
    fail,
    help_callback,
    log_parameters,
    write_umap,
)
from glmpca.LSA import LSA

DESCRIPTION = (
    "Reduce the dimensionality of a cell-by-feature count matrix with latent "
    "semantic analysis (LSA).\n\n"
    "``scLSA`` weighs the counts in ``.X`` of an h5ad file, with cells in rows and "
    "features in columns, and computes the truncated SVD of the weighted matrix. "
    "The result is an updated h5ad object with:\n"
    '* ``obsm["X_lsa"]``: the coordinates of the cells.\n'
    '* ``varm["LSA_loadings"]``: the loadings of the features.\n'
    '* ``uns["LSA"]``: the parameters, the singular values and, with '
    "``--dropDepth``, the components dropped.\n\n"
    "If ``--outFileUMAP`` is given, ``scLSA`` also computes a 2D projection (UMAP) "
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
_LSA = "LSA options"
_CLUSTERING = "Clustering options"
_PLOT = "Plot options"
_OTHER = "Other options"


class Flavor(str, Enum):
    signac = "signac"
    gensim = "gensim"


class Solver(str, Enum):
    auto = "auto"
    arpack = "arpack"
    randomized = "randomized"


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"scLSA (glmpca {version('glmpca')})")
        raise typer.Exit()


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
            help=("The input .h5ad file."),
        ),
    ],
    out_file: Annotated[
        str,
        typer.Option(
            "-o",
            "--outFile",
            metavar=".h5ad",
            rich_help_panel=_IO,
            help="The output .h5ad file: the input with the LSA results added.",
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
    # LSA options
    n_prin_comps: Annotated[
        int,
        typer.Option(
            "-n",
            "--nPrinComps",
            min=1,
            rich_help_panel=_LSA,
            help=(
                "Number of components to reduce the dimensionality to. Use a higher "
                "number for samples with more expected heterogeneity."
            ),
        ),
    ] = 20,
    flavor: Annotated[
        Flavor,
        typer.Option(
            "--flavor",
            metavar="FLAVOR",
            rich_help_panel=_LSA,
            help=(
                "The weighting of the counts.\n\n"
                "[bold yellow]signac[/bold yellow]: "
                "``log1p(count / depth · cells / feature total · 10,000)``"
                "and standardized scores.\n\n"
                "[bold yellow]gensim[/bold yellow]: gensim's "
                '``TfidfModel(smartirs="lfu")``.'
            ),
        ),
    ] = Flavor.signac,
    solver: Annotated[
        Solver,
        typer.Option(
            "--solver",
            metavar="SOLVER",
            rich_help_panel=_LSA,
            help=(
                "The SVD solver.\n\n"
                "[bold yellow]arpack[/bold yellow]: the exact truncated SVD of "
                "``scipy.sparse.linalg.svds``, on the CPU, with the matrix in memory."
                "\n\n"
                "[bold yellow]randomized[/bold yellow]: the randomized SVD of Halko et "
                "al., which reads the matrix by blocks of cells, ``3 + 2 x --nIter`` "
                "times, on the CPU or the GPU, from memory or with ``--backed``.\n\n"
                "[bold yellow]auto[/bold yellow]: arpack for a matrix in memory on the "
                "CPU, else randomized."
            ),
        ),
    ] = Solver.auto,
    n_iter: Annotated[
        int,
        typer.Option(
            "--nIter",
            min=0,
            rich_help_panel=_LSA,
            help=(
                "Power iterations of the randomized solver. More iterations give "
                "components nearer to the exact ones, and each one costs two passes "
                "over the matrix."
            ),
        ),
    ] = 4,
    drop_depth: Annotated[
        bool,
        typer.Option(
            "--dropDepth",
            rich_help_panel=_LSA,
            help=(
                "Drop every component whose scores follow the depth of the cells "
                "(|Spearman rho| above 0.75), off by default, so fewer than "
                "``--nPrinComps`` components can be left. The number dropped is "
                'reported and stored in ``uns["LSA"]``.'
            ),
        ),
    ] = False,
    binarize: Annotated[
        bool,
        typer.Option(
            "--binarize",
            rich_help_panel=_LSA,
            help=(
                "Read every value that is not zero as 1, off by default. The output "
                "file keeps the original matrix in ``.X``."
            ),
        ),
    ] = False,
    backed: Annotated[
        bool,
        typer.Option(
            "--backed",
            rich_help_panel=_LSA,
            help=(
                "Open the input in backed mode and read the matrix from the file by "
                "blocks of cells, instead of loading it into memory, so the matrix "
                "can be larger than the memory. Each pass reads the file again. It "
                "needs the randomized solver."
            ),
        ),
    ] = False,
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
    ] = 15,
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
                "The device of the randomized solver, for example "
                "[bold yellow]cpu[/bold yellow], [bold yellow]cuda[/bold yellow] or "
                "[bold yellow]cuda:1[/bold yellow]."
            ),
        ),
    ] = None,
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
            out_file_umap=out_file_umap,
            n_prin_comps=n_prin_comps,
            flavor=flavor,
            solver=solver,
            n_iter=n_iter,
            drop_depth=drop_depth,
            binarize=binarize,
            backed=backed,
            n_neighbors=n_neighbors,
            cluster_resolution=cluster_resolution,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_file_format=plot_file_format,
            dpi=dpi,
            device=device,
        )
    else:
        warnings.filterwarnings("ignore")

    if backed and solver is Solver.arpack:
        msg = (
            "--solver arpack needs the matrix in memory, so it cannot be used with "
            "--backed. Use --solver randomized."
        )
        raise fail(msg)
    adata = ad.read_h5ad(input, backed="r") if backed else ad.read_h5ad(input)
    if adata.X is None:
        msg = f"'{input}' has no matrix in .X."
        raise fail(msg)

    model = LSA(
        n_pc=n_prin_comps,
        flavor=flavor.value,
        solver=solver.value,
        n_iter=n_iter,
        drop_depth=drop_depth,
        binarize=binarize,
        device=device,
    )
    try:
        scores = model.fit_transform(adata).numpy()
    except ValueError as exc:
        raise fail(str(exc)) from exc
    assert model.components_ is not None
    assert model.singular_values_ is not None

    adata.obsm["X_lsa"] = scores
    adata.varm["LSA_loadings"] = model.components_.numpy()
    adata.uns["LSA"] = {
        "params": {
            "n_pc": n_prin_comps,
            "flavor": flavor.value,
            "solver": solver.value,
            "solver_used": model.solver_,
            "n_iter": n_iter,
            "drop_depth": drop_depth,
            "binarize": binarize,
            "backed": backed,
            "random_state": model.random_state,
        },
        "singular_values": model.singular_values_.numpy(),
    }
    if model.depth_correlations_ is not None:
        adata.uns["LSA"]["depth_correlations"] = model.depth_correlations_
        adata.uns["LSA"]["n_dropped_components"] = model.n_dropped_components_
    logging.info("components kept: %d", scores.shape[1])

    if out_file_umap is not None:
        write_umap(
            adata,
            scores,
            out_file_umap,
            n_neighbors,
            cluster_resolution,
            plot_width,
            plot_height,
            plot_file_format,
            dpi,
        )

    adata.write_h5ad(out_file)

    return 0


def cli() -> None:
    configure_logging()
    app()


if __name__ == "__main__":
    cli()
