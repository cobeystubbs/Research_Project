# Importing the necessary packages
import scanpy as sc
import scanpy as sc
import pandas as pd

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
pp.highly_variable_genes(sc_data, n_top_genes=2000, batch_key="sample")
sc.pl.highly_variable_genes(sc_data)

# Dimensionality reduction by running PCA
sc.tl.pca(adata)
# Inspect contribution of single PCs to inform how many to consider when computing the neighbourhood relation of cells
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)

# Plotting PCA
sc.pl.pca(
    adata,
    color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=2,
)




