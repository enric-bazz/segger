reference_spec = self._get_reference_raster_spec(
    input_sdata,
    reference_image_key,
)

points_elements = self._build_points_element(...)

shapes_elements = self._build_shapes_element(...)

labels_elements = self._build_labels_element(
    reference_spec=reference_spec,
    shapes=shapes,
)

tables_elements = self._build_table_element(
    region="segmentation",
)

return SpatialData(
    points=points_elements,
    shapes=shapes_elements,
    labels=labels_elements,
    tables=tables_elements,
)