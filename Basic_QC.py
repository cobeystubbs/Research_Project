# Importing the necessary packages
import anndata as ad
import scanpy as sc
import pandas as pd
import celltypist as ct
from matplotlib import rc_context

# Load in the data
# Starting with the 10X genomics MTX data
sc_data = sc.read_10x_mtx("/Users/cobeystubbs/Downloads/WBC011/filtered_feature_bc_matrix", var_names='gene_symbols')
print(sc_data)

meta = pd.read_csv("/Users/cobeystubbs/Downloads/WBC011/per_barcode_metrics.csv")

# mitochondrial genes, "MT-" for human, "Mt-" for mouse
sc_data.var["mt"] = sc_data.var_names.str.startswith("MT-")
# ribosomal genes
sc_data.var["ribo"] = sc_data.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
sc_data.var["hb"] = sc_data.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(sc_data, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

# Creating the violin plots
# Number of genes by counts, total counts, and the percentage of counts which are mitochondrial
sc.pl.violin(
    sc_data,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.3,
    multi_panel=True,
)

# Creating a QC scatter plot
sc.pl.scatter(sc_data, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

# Filtering using default parameters
sc.pp.filter_cells(sc_data, min_genes=100)
sc.pp.filter_genes(sc_data, min_cells=3)

# Using scrublet to quality control filter out doublets
sc.pp.scrublet(sc_data, expected_doublet_rate=0.06)

# Saving the count data for normalisation
sc_data.layers["counts"] = sc_data.X.copy()
# Normalizing to median total counts
sc.pp.normalize_total(sc_data)
# Logarithmise the data
sc.pp.log1p(sc_data)

# Feature selection
sc.pp.highly_variable_genes(sc_data, n_top_genes=2000)
sc.pl.highly_variable_genes(sc_data)

# Dimensionality reduction by running PCA
sc.tl.pca(sc_data)
# Inspect contribution of single PCs to inform how many to consider when computing the neighbourhood relation of cells
sc.pl.pca_variance_ratio(sc_data, n_pcs=50, log=True)

# Completing the neighbourhood graph
sc.pp.neighbors(sc_data)
sc.tl.umap(sc_data)
# Plotting the umap
sc.pl.umap(
    sc_data,
    size=2,
)

# Using the igraph implementation and a fixed number of iterations can be significantly faster,
# especially for larger datasets
sc.tl.leiden(sc_data, flavor="igraph", n_iterations=2)
sc.pl.umap(sc_data, color=["leiden"])

sc.pl.umap(
    sc_data,
    color=["leiden", "predicted_doublet", "doublet_score"],
    wspace=0.5,
    size=3,
)

sc.pl.umap(
    sc_data,
    color=["leiden", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
    wspace=0.5,
    ncols=2,
)

# Filtering out the predicted doublets
sc_data = sc_data[sc_data.obs['predicted_doublet'] == False].copy()

# QC to ensure that the doublets have been removed
sc.pl.umap(
    sc_data,
    color=["leiden", "predicted_doublet", "doublet_score"],
    # increase horizontal space between panels
    wspace=0.5,
    size=3
)

# Look at different resolutions of UMAP clustering
for res in [0.02, 0.35, 2.0]:
    sc.tl.leiden(sc_data, key_added=f"leiden_res_{res:4.2f}", resolution=res, flavor="igraph")
sc.pl.umap(
    sc_data,
    color=["leiden_res_0.02", "leiden_res_0.35", "leiden_res_2.00"],
    legend_loc="on data",
)

for res in [0.5]:
    sc.tl.leiden(sc_data, key_added=f"leiden_res_{res:4.2f}", resolution=res, flavor="igraph")
sc.pl.umap(
    sc_data,
    color=["leiden_res_0.50"],
    legend_loc="on data",
)

marker_genes = {
    "CD14+ Mono": ["FCN1", "CD14", "LYZ"],
    "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN", "CD16", "FCGR3A", "MS4A7"],
    # Note: DMXL2 should be negative
    "cDC2": ["CST3", "COTL1", "LYZ", "DMXL2", "CLEC10A", "FCER1A", "CD1C"],
    "Erythroblast": ["MKI67", "HBA1", "HBB", "HBA2"],
    # Note HBM and GYPA are negative markers
    "Proerythroblast": ["CDK6", "SYNGR1", "HBM", "GYPA", "KLF1", "GATA1"],
    "NK": ["GNLY", "NKG7", "CD247", "FCER1G", "TYROBP", "KLRG1", "FCGR3A", "KLRD1", "PRF1"],
    "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1", "IL7R", "CD127", "TNFRSF18"],
    "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM", "CD19"],
    # Note IGHD and IGHM are negative markers
    "B cells": [
        "MS4A1",
        "ITGB1",
        "COL4A4",
        "PRDM1",
        "IRF4",
        "PAX5",
        "BCL11A",
        "BLK",
        "IGHD",
        "IGHM",
        "CD79A"
        "CD79B"
        "BANK1"
    ],
    "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN", "SDC1"],
    # Note PAX5 is a negative marker
    "Plasmablast": ["XBP1", "PRDM1", "PAX5", "CD38"],
    "CD4+ T": ["CD4", "IL7R", "TRBC2"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH", "GZMA"],
    "T naive": ["LEF1", "CCR7", "TCF7", "IL7R", "CCR7"],
    "T cell": ["CD69", "HLA-DRA", "IL2RA", "CD25"],
    "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4", "LILRA4", "TCF4"],
}

# Filter marker_genes to only include genes in sc_data
marker_genes_filtered = {
    ct: [g for g in genes if g in sc_data.var_names]
    for ct, genes in marker_genes.items()
}
# Remove any cell types with no remaining genes
marker_genes_filtered = {ct: genes for ct, genes in marker_genes_filtered.items() if genes}

sc.pl.dotplot(sc_data, marker_genes_filtered, groupby="leiden_res_0.50", standard_scale="var")

# Obtain cluster-specific differentially expressed genes
# Remove cells in clusters with fewer than 2 cells
cluster_counts = sc_data.obs["leiden_res_0.50"].value_counts()
keep_clusters = cluster_counts[cluster_counts >= 2].index
sc_data_filtered = sc_data[sc_data.obs["leiden_res_0.50"].isin(keep_clusters)].copy()

sc.tl.rank_genes_groups(sc_data_filtered, groupby="leiden_res_0.50", method="wilcoxon")
sc.pl.rank_genes_groups_dotplot(sc_data_filtered, groupby="leiden_res_0.50", standard_scale="var", n_genes=5)
sc.pl.rank_genes_groups_dotplot(sc_data_filtered, groupby="leiden_res_0.50", standard_scale="var", n_genes=15)


sc_data.obs["cell_type_lvl1"] = sc_data.obs["leiden_res_0.50"].map(
    {
        "0": "B cells",
        "1": "?",
        "2": "NK cells",
        "3": "?",
        "4": "pDC cells",
        "5": "monocytes",
        "6": "dendritic cells",
        "7": "mitochondrial cluster",
        "8": "naive CD20+ B cells",
        "9": "T cells",
    }
)

sc.pl.umap(
    sc_data,
    color=["cell_type_lvl1"],
    # increase horizontal space between panels
    wspace=0.5,
    size=10,
)

# Using celltypist to predict the cell types and annotate clusters
predictions = ct.annotate(sc_data, majority_voting = False)
predictions_adata = predictions.to_adata()
sc.pl.umap(predictions_adata, color="predicted_labels")

color_vars = [
    "TFAM"
]
with rc_context({"figure.figsize": (3, 3)}):
    sc.pl.umap(sc_data, color=color_vars, s=50, frameon=False, ncols=4, vmax="p99",
               save="umap_tfam.svg")



# pD1 marker
# plot on umap the mt genes + see if they pop up everywhere (they should)
# can we create a mitochondrial score from those genes that mimics the amount of mitochondria inside (mean, median expression of genes?)
# some are always there - expect robust
# mitocarta
