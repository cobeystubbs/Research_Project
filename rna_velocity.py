# =====source /Users/cobeystubbs/scvelo_env/bin/activate=====
# Python version 3.12.13

import scvelo as scv
import scanpy as sc

scv.logging.print_version()

scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.presenter_view = True  # set max width size for presenter view
scv.set_figure_params('scvelo')  # for beautified visualization

adata = sc.read('/Users/cobeystubbs/Downloads/adata.loom')
adata.layers['unspliced'] = adata.layers['nascent']
adata.layers['spliced'] = adata.layers['mature']

print(adata)

print(adata.layers.keys())
adata_filtered = adata.copy()

import scanpy as sc
print(adata)
print(adata.layers.keys())
scv.pl.proportions(adata)

import numpy as np

print("Total spliced:", np.sum(adata.layers["spliced"]))
print("Total unspliced:", np.sum(adata.layers["unspliced"]))
cell_counts = np.sum(adata.layers["spliced"] + adata.layers["unspliced"], axis=1)
print(np.sum(cell_counts == 0))

adata_filtered = adata.copy()
scv.pp.filter_genes(adata_filtered, min_shared_counts=1)
print(adata_filtered)