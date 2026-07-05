from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio.features
import xarray as xr

from spatialdata import SpatialData
from spatialdata.models import Labels2DModel


def rasterize_shapes_to_labels(
    sdata: SpatialData,
    shapes_key: str,
    reference_image_key: str,
    output_labels_key: Optional[str] = None,
    scale: str = "scale0",
    background: int = 0,
    dtype: np.dtype = np.uint32,
):
    """
    Rasterize a SpatialData shapes element into a labels element aligned to
    a reference image grid.

    Parameters
    ----------
    sdata
        SpatialData object.

    shapes_key
        Key of the shapes element in `sdata.shapes`.

    reference_image_key
        Key of the reference image in `sdata.images`.

    output_labels_key
        Optional key to store the generated labels in `sdata.labels`.
        If None, labels are not added automatically.

    scale
        Multiscale image level to rasterize against.

    background
        Background label value.

    dtype
        Output dtype for labels.

    Returns
    -------
    labels
        Parsed Labels2DModel object.

    Notes
    -----
    Assumes shapes and reference image are already expressed in the same
    coordinate system.
    """

    gdf: gpd.GeoDataFrame = sdata.shapes[shapes_key]

    img = sdata.images[reference_image_key]

    # handle multiscale images
    if hasattr(img, "__getitem__") and scale in img:
        img_scale = img[scale]
    else:
        img_scale = img

    height = img_scale.sizes["y"]
    width = img_scale.sizes["x"]

    try:
        transform = img_scale.rio.transform()
    except Exception as e:
        raise RuntimeError(
            "Could not retrieve affine transform from reference image."
        ) from e

    # assign unique integer labels starting from 1
    shapes_iter = (
        (geom, idx)
        for idx, geom in enumerate(gdf.geometry, start=1)
        if geom is not None and not geom.is_empty
    )

    labels_np = rasterio.features.rasterize(
        shapes=shapes_iter,
        out_shape=(height, width),
        transform=transform,
        fill=background,
        dtype=dtype,
    )

    labels_xr = xr.DataArray(
        labels_np,
        dims=("y", "x"),
        coords={
            "y": img_scale.coords["y"],
            "x": img_scale.coords["x"],
        },
    )

    labels = Labels2DModel.parse(labels_xr)

    if output_labels_key is not None:
        sdata.labels[output_labels_key] = labels

    return labels