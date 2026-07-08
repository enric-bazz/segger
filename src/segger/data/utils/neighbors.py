from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from typing import Any, Literal
import geopandas as gpd
import polars as pl
import numpy as np
import cupy as cp
import cugraph
import torch
import cuml
import cudf
import gc
import warnings

from ...io import TrainingTranscriptFields, TrainingBoundaryFields
from ...geometry import points_in_polygons


def phenograph_rapids(
    X: ArrayLike,
    n_neighbors: int,
    min_size: int = -1,
    **kwargs,
) -> np.ndarray:
    """TODO: Add description.
    """
    X = cp.array(X)
    model = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(X)
    _, indices = model.kneighbors(X)

    n, k = indices.shape
    edges = cudf.concat([
        cudf.Series(np.repeat(np.arange(n), k), name='source', dtype="int32"),
        cudf.Series(indices.flatten(), name='destination', dtype="int32"),
    ], axis=1)
    G = cugraph.from_cudf_edgelist(edges)
    
    # Build jaccard-weighted graph in GPU
    jaccard_edges = cugraph.jaccard(G, edges[['source', 'destination']])
    G = cugraph.from_cudf_edgelist(jaccard_edges, *jaccard_edges.columns)
    
    # Cluster jaccard-weighted graph
    result, _ = cugraph.louvain(G, **kwargs)
    
    # Sort clusters by size
    sizes = result['partition'].value_counts()
    sizes.loc[:] = cp.where(sizes > min_size, cp.arange(len(sizes)), -1)
    result['partition'] = result['partition'].map(sizes)
    
    # Sort by vertex (e.g. cell)
    df = result.sort_values('vertex')
    col = df['partition']
    print(type(col))
    print(col.__cuda_array_interface__)
    vals = col.to_arrow().to_numpy()

    return vals


def knn_to_edge_index(
    neighbor_table: torch.Tensor,
    padding_value = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dense neighbor table (with padding) into COO edge index.

    Parameters
    ----------
    neighbor_table : (N, K) long tensor
        Sampled neighbor table with N used as padding value.

    Returns
    -------
    edge index : (2, E) long tensor
    index pointer  : (N+1,) long tensor
    """
    with torch.no_grad():
        N, K   = neighbor_table.shape
        if padding_value is None:
            padding_value = N
        device = neighbor_table.device

        valid  = neighbor_table != padding_value
        flat   = valid.view(-1).nonzero(as_tuple=False).squeeze(1)
        col    = neighbor_table.view(-1)[flat]
        row    = flat // K

        edge_index = torch.stack([row, col])

        deg = valid.sum(dim=1)
        index_ptr = torch.cat(
            (torch.zeros(1, dtype=torch.long, device=device), deg.cumsum(0))
        )
        del valid, flat, col, row, deg
        torch.cuda.empty_cache()
        gc.collect()

    return edge_index, index_ptr


def edge_index_to_knn(
    edge_index: torch.Tensor,
    padding_value: Any = None,
) -> torch.Tensor:
    """TODO: Add description.
    """
    _, lengths = torch.unique_consecutive(
        edge_index[0],
        return_counts=True,
    )
    B = lengths.size(0)
    L = lengths.max()
    neighbor_table = edge_index[0].new_full((B, L), -1)
    
    row = torch.repeat_interleave(
        torch.arange(B, device=neighbor_table.device),
        lengths
    )
    start = torch.cumsum(lengths, 0) - lengths
    col = torch.arange(edge_index[0].size(0), device=neighbor_table.device)
    col -= torch.repeat_interleave(start, lengths)

    neighbor_table[row, col] = edge_index[1]

    return neighbor_table


def kdtree_neighbors(
    points: np.ndarray,
    max_k: int,
    max_dist: float = np.inf,
    query: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for KDTree kNN and conversion to edge_index COO format.
    TODO: Add description.
    """
    tree = KDTree(points, leafsize=100)
    _, indices = tree.query(
        points if query is None else query,
        k=max_k,
        distance_upper_bound=max_dist,
        workers=-1,
    )
    indices = torch.from_numpy(indices)
    gc.collect()  # make sure numpy copy is gone before conversion
    edge_index, index_pointer = knn_to_edge_index(indices)
    del indices   # remove big indices tensor
    gc.collect()
    
    return edge_index, index_pointer


def setup_transcripts_graph(
    tx: pl.DataFrame,
    max_k: int,
    max_dist: float,
    use_3d: bool | Literal["auto"] = False,
) -> torch.Tensor:
    """TODO: Add description.
    """
    tx_fields = TrainingTranscriptFields()
    coord_cols = [tx_fields.x, tx_fields.y]
    has_z = tx_fields.z in tx.columns

    if use_3d == "auto":
        use_3d = has_z and tx[tx_fields.z].null_count() < len(tx)
    elif use_3d is True and not has_z:
        warnings.warn(
            f"use_3d=True but z column '{tx_fields.z}' not found in transcripts. "
            "Falling back to 2D graph construction.",
            RuntimeWarning,
            stacklevel=2,
        )
        use_3d = False
    if use_3d and has_z:
        coord_cols.append(tx_fields.z)

    points = tx[coord_cols].to_numpy()
    edge_index, _ = kdtree_neighbors(
        points=points,
        max_k=max_k,
        max_dist=max_dist,
    )
    return edge_index


def setup_segmentation_graph(
    tx: pl.DataFrame,
    segmentation_mask: pl.Expr | pl.Series = None,
) -> torch.Tensor:
    """TODO: Add description.
    """
    tx_fields = TrainingTranscriptFields()
    return (
        tx
        .with_row_index("_tid")
        .filter(segmentation_mask)
        .select(["_tid", tx_fields.cell_encoding])
        .to_torch()
        .T
    )


def setup_prediction_graph(
    tx: pl.DataFrame,
    bd: gpd.GeoDataFrame,
    max_k: int,
    scale_factor: float,
    mode: Literal['nucleus', 'cell', 'uniform'] = 'cell',
    use_3d: bool | Literal["auto"] = False,
) -> torch.Tensor:
    """TODO: Add description.
    """
    tx_fields = TrainingTranscriptFields()
    bd_fields = TrainingBoundaryFields()

    # Uniform kNN graph
    if mode == "uniform":
        coord_cols = [tx_fields.x, tx_fields.y]
        has_z = tx_fields.z in tx.columns
        if use_3d == "auto":
            use_3d = has_z and tx[tx_fields.z].null_count() < len(tx)
        elif use_3d is True and not has_z:
            warnings.warn(
                f"use_3d=True but z column '{tx_fields.z}' not found in transcripts. "
                "Falling back to 2D graph construction.",
                RuntimeWarning,
                stacklevel=2,
            )
            use_3d = False
        if use_3d and has_z:
            coord_cols.append(tx_fields.z)

        points = tx[coord_cols].to_numpy()
        query = bd.geometry.centroid.get_coordinates().values
        if use_3d and len(coord_cols) == 3:
            query = np.hstack([query, np.zeros((len(query), 1))])
        edge_index, _ = kdtree_neighbors(
            points=points,
            query=query,
            max_k=max_k,
            max_dist=np.inf,
        )
        return edge_index
    
    # Shape-based graph
    from shapely.affinity import scale as shapely_scale

    points = tx[[tx_fields.x, tx_fields.y]].to_numpy()
    boundary_type = (bd_fields.cell_value if mode == "cell"
                     else bd_fields.nucleus_value)
    polygons = bd[bd[bd_fields.boundary_type] == boundary_type].geometry
    if polygons.empty:
        warnings.warn(
            f"No '{boundary_type}' boundaries found for prediction graph. "
            "Falling back to all available boundaries.",
            RuntimeWarning,
            stacklevel=2,
        )
        polygons = bd.geometry
    if polygons.empty:
        raise ValueError(
            "Prediction graph construction requires at least one boundary polygon."
        )

    # Multiplicative polygon scaling around centroid.
    # scale_factor > 1 expands, < 1 shrinks.
    polygons = polygons.apply(
        lambda geom: shapely_scale(
            geom,
            xfact=scale_factor,
            yfact=scale_factor,
            origin="centroid",
        )
    ).reset_index(drop=True)

    result = points_in_polygons(
        points=points,
        polygons=polygons,
        predicate='contains',
        batches=10,
    )

    return torch.tensor(
        result[['index_query', 'index_match']].values.T).to(torch.int).cpu()
