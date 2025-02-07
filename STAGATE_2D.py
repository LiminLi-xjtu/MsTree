import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys

import torch
import STAGATE_pyG


#Preparation
counts = pd.read_csv('/data/simulation/gene10000/sp_sim_count_ieffect5.csv', index_col=0)
counts = counts.T
coor_df = pd.read_csv('/data/simulation/gene10000/sp_sim_location.csv', index_col=0)
print(counts.shape, coor_df.shape)


adata = sc.AnnData(counts,obs=coor_df)
coor_df = coor_df.loc[adata.obs_names, ['x', 'y']]
adata.obsm["spatial"] = coor_df.to_numpy()


#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print(adata)


#Constructing the spatial network
STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=0.04) ##n=2000
STAGATE_pyG.Stats_Spatial_Net(adata)


#Running STAGATE
adata = STAGATE_pyG.train_STAGATE(adata, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

sc.pp.neighbors(adata, use_rep='STAGATE')
sc.tl.umap(adata)

print(adata)
temp = pd.DataFrame(adata.obsm['STAGATE'])
print(temp)
temp.to_csv('/data/simulation/gene10000/sim_ieffect5_STAGATE_pyG.csv')


