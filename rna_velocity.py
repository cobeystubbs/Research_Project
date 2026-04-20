# =====source /Users/cobeystubbs/scvelo_env/bin/activate=====
# Python version 3.12.13

import scvelo as scv
import scanpy as sc

def main():
    scv.logging.print_version()

    scv.settings.verbosity = 3
    scv.settings.presenter_view = True
    scv.set_figure_params('scvelo')

    adata = sc.read('/Users/cobeystubbs/Downloads/WBC011.loom')

    scv.pl.proportions(adata)

    scv.pp.filter_and_normalize(adata)
    scv.pp.moments(adata)

    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    scv.pl.velocity_embedding_stream(adata, basis="umap")

if __name__ == "__main__":
    main()