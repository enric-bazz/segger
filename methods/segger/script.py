import anndata as ad
import numpy as np
import os
import shutil
import spatialdata as sd
import xarray as xr
from spatialdata.models import Labels2DModel
import torch

from rasterio.features import rasterize

from lightning.pytorch import Trainer


from segger.data import ISTDataModule

from segger.models import LitISTEncoder

# from segger.data import ISTSegmentationWriter
from segger.openproblems_utils.utils import rasterize_shapes_to_labels

from segger.export.spatialdata_writer import _create_spatialdata, _merge_predictions

from segger.io.preprocessor import ISTPreprocessor

# use assign_transcitps_to_boundaries to do the spatial join with boundaries.




## VIASH START
par = {
  'input': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_unlabelled.zarr',
  'output': 'prediction.zarr'
}
meta = {
  'name': 'segger'
}
## VIASH END


print('Reading input', flush=True)
in_sdata = sd.read_zarr(par["input"])

#### SEGGER code

### Segger paramters
cells_representation = "pca"
node_representation_dim = 128
cells_min_counts = 10
cells_clusters_n_neighbors = 10
cells_clusters_resolution = 2
genes_min_counts = 100
genes_clusters_n_neighbors = 5
genes_clusters_resolution = 2.
transcripts_max_k = 5
transcripts_max_dist = 5.
segmentation_graph_negative_edge_rate = 1.
prediction_mode = "cell"
prediction_max_k = 3
prediction_expansion_ratio = 0.05
tiling_mode = "adaptive"  # TODO: Remove (benchmarking only)
tiling_margin_training = 20.
tiling_margin_prediction = 20.
max_nodes_per_tile = 50_000
tiling_side_length = 250.  # TODO: Remove (benchmarking only)
training_fraction = 0.75
max_edges_per_batch = 1_000_000
gene_corr_reference_path = None 
gene_missing_strategy = "error"


use_3d = True
min_qv = 20

# DataModule parameters

n_epochs = 20


# Setup Lightning Data Module
print("Reading spatialdata input with Segger APIs and setting up ISTDataModule", flush=True)

datamodule = ISTDataModule(
    input_directory=par["input"],
    cells_representation_mode=cells_representation,
    cells_embedding_size=node_representation_dim,
    cells_min_counts=cells_min_counts,
    cells_clusters_n_neighbors=cells_clusters_n_neighbors,
    cells_clusters_resolution=cells_clusters_resolution,
    genes_clusters_n_neighbors=genes_clusters_n_neighbors,
    genes_clusters_resolution=genes_clusters_resolution,
    transcripts_graph_max_k=transcripts_max_k,
    transcripts_graph_max_dist=transcripts_max_dist,
    prediction_graph_mode=prediction_mode,
    prediction_graph_max_k=prediction_max_k,
    prediction_graph_buffer_ratio=prediction_expansion_ratio,
    tiling_margin_training=tiling_margin_training,
    tiling_margin_prediction=tiling_margin_prediction,
    tiling_nodes_per_tile=max_nodes_per_tile,
    edges_per_batch=max_edges_per_batch,
    use_3d=use_3d,
    min_qv=min_qv,
)

# Setup Lightning Model

print("Setting up LitISTEncoder model", flush=True)

n_genes = datamodule.ad.shape[1]
segmentation_loss = "triplet"
hidden_channels: int = 64,
out_channels: int = 64,
n_mid_layers: int = 2,
n_heads: int = 2,
learning_rate: float = 1e-3,
sg_loss_type: str = 'triplet',
transcripts_margin = 0.3,
segmentation_margin = 0.4,
transcripts_loss_weight_start = 1.,
transcripts_loss_weight_end = 1.,
cells_loss_weight_start = 1.,
cells_loss_weight_end = 1.,
segmentation_loss_weight_start = 0.,
segmentation_loss_weight_end = 0.5,
update_gene_embedding: bool = True,
use_positional_embeddings: bool = True,
normalize_embeddings: bool = True,



model = LitISTEncoder(
    n_genes=n_genes,
    n_mid_layers=n_mid_layers,
    n_heads=n_heads,
    in_channels=node_representation_dim,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    learning_rate=learning_rate,
    sg_loss_type=segmentation_loss,
    tx_margin=transcripts_margin,
    sg_margin=segmentation_margin,
    tx_weight_start=transcripts_loss_weight_start,
    tx_weight_end=transcripts_loss_weight_end,
    bd_weight_start=cells_loss_weight_start,
    bd_weight_end=cells_loss_weight_end,
    sg_weight_start=segmentation_loss_weight_start,
    sg_weight_end=segmentation_loss_weight_end,
    normalize_embeddings=normalize_embeddings,
    use_positional_embeddings=use_positional_embeddings,
)


# Disable logging, set to True to enable TensorBoard logging or pass original csv logger (below)
logger = False
# from lightning.pytorch.loggers import CSVLogger
# logger = CSVLogger(par["output"])

trainer = Trainer(
    logger=logger,
    max_epochs=n_epochs,
    reload_dataloaders_every_n_epochs=1,
)

# Training
print("Starting training", flush=True)
trainer.fit(model=model, datamodule=datamodule)

# Prediction
print("Predicting segmentation", flush=True)
predictions = trainer.predict(model=model, datamodule=datamodule)

print('Segger segmentation finished, post-processing results', flush=True)
# create anndata table and run boundaries computation


print('Creating output data structure', flush=True)
# create spatialdata obj



print('Creating output data structure', flush=True)

tx = trainer.datamodule.tx

# Merge predictions with transcripts
merged = _merge_predictions(
	predictions=predictions,
	transcripts=transcripts,
	row_index_column=row_index_column,
	cell_id_column=cell_id_column,
	similarity_column=similarity_column,
)

sd_output = _create_spatialdata(
	transcripts=tx,
	x_column='x',
	y_column='y',
	z_column='z' if use_3d else None,
	cell_id_column="cell_id",
	feature_column='feature_id',
)

mask = rasterize_shapes_to_labels(
    sdata=sd_output,
	shapes_key="segmentation",
	reference_image_key=in_sdata['morphology_mip']['scale0'].image,
	scale='scale0',
	background=0,
	dtype=np.uint16,
    
)
	sd_output = sd.SpatialData(
	labels={
		'segmentation': Labels2DModel.parse(
		xr.DataArray(masks, name='segmentation', dims=('y', 'x')),
		transformations=transformation
		)
	},
	tables={
		'table': ad.AnnData(
		uns={
			'dataset_id': sdata.tables['table'].uns['dataset_id'],
			'method_id': meta['name']
		}
		)
	}
)


print('Saving output', flush=True)
if os.path.exists(par["output"]):
    shutil.rmtree(par["output"])
sd_output.write(par["output"])

