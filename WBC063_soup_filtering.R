library(SoupX)
library(Seurat)
library(SeuratDisk)

sc = load10X("/Users/cobeystubbs/Desktop/outs")

# Create Seurat object
seu <- CreateSeuratObject(sc$toc)

# Standard preprocessing
seu <- NormalizeData(seu)
seu <- FindVariableFeatures(seu)
seu <- ScaleData(seu)
seu <- RunPCA(seu)
seu <- FindNeighbors(seu)
seu <- FindClusters(seu)

# Give SoupX the cluster labels
sc = setClusters(sc, Idents(seu))
# Estimate rho
sc = autoEstCont(sc)
# Clean the data
out = adjustCounts(sc)

plotChangeMap(sc, out, "IGKC")

DropletUtils:::write10xCounts("/Users/cobeystubbs/Desktop/soupx_filtered_feature_bc_matrix", out)

