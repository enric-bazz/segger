# Segger Validation Metrics Reference

This document describes the 11 quality metrics computed by `segger validate`. All implementations live in `src/segger/validation/quick_metrics.py`.

## Overview

The validation suite evaluates cell segmentation quality across four axes:

| Axis | Metrics |
|------|---------|
| **Coverage** | Transcript Assignment Coverage |
| **Specificity** | MECR, RESOLVI Contamination, Spurious Coexpression, Border Contamination, Center-Border Similarity |
| **Sensitivity** | Positive Marker Recall |
| **Morphology & Spatial** | Reference Morphology Match, Transcript-Centroid Offset, Signal Doublet, Vertical Doublet |

### Reporting Names (Plots/Dashboard)

The following display names are used in tradeoff/resource reporting.  
CLI flags and output key names remain unchanged.

| Display Name | Legacy Label | CLI Flag / Key Family |
|--------------|--------------|-----------------------|
| Center-Border Similarity | Center-Border NCV | `--center-border-ncv`, `center_border_ncv_*` |
| Transcript-Centroid Offset | TCO | `--tco`, `transcript_centroid_offset_fast` |
| Vertical Doublet | VSI Doublet / Signal Hotspot Doublet | `--signal-hotspot` / `--vsi`, `signal_hotspot_doublet_*` |

### How Segmentation Parameters Affect Metrics

These metrics are computed on the *output* of `segment` or `predict`. The parameter choices made upstream directly shape which metrics improve or degrade:

| Parameter | Primary Metric Impact |
|-----------|----------------------|
| `--prediction-scale-factor` ↑ | Assignment ↑, MECR ↑, TCO ↓ |
| `--fragment-mode` on | Assignment ↑↑, MECR ↑↑, Doublet ↑ |
| `--alignment-loss-weight-end` ↑ | MECR ↓, Assignment risk ↓ at high values |
| `--use-3d true` | Doublet ↓ (modest), other metrics ±1% |
| `--min-similarity-shift` ↑ | Assignment ↑, MECR ↑ (more permissive thresholds) |

See [SEGMENT.md](SEGMENT.md#empirical-parameter-guide) for detailed empirical analysis of these tradeoffs.

### CLI Usage

```bash
# Run all metrics
segger validate -s segger_segmentation.parquet -i /path/to/source -a segger_segmentation.h5ad \
    --scrna-reference-path reference.h5ad

# Run specific metrics
segger validate -s segger_segmentation.parquet --assigned --mecr -a segger_segmentation.h5ad

# Auto-fetch scRNA reference by tissue type
segger validate -s segger_segmentation.parquet -i /path/to/source \
    --tissue-type colon -a segger_segmentation.h5ad
```

If no metric flags are provided, all metrics run. If any flag is set, only selected metrics run.

### Common Inputs

| Input | CLI Flag | Description |
|-------|----------|-------------|
| Segmentation parquet | `-s` / `--segmentation-path` | Required. Must contain `row_index` and `segger_cell_id` columns. |
| Source data | `-i` / `--source-path` | Raw platform data (Xenium/MERSCOPE/CosMX) or SpatialData `.zarr`. Needed for source-based metrics. |
| AnnData | `-a` / `--anndata-path` | `segger_segmentation.h5ad` for MECR. |
| scRNA reference | `--scrna-reference-path` | `.h5ad` with cell type annotations. Used by MECR discovery, Marker Recall, RESOLVI. |
| Tissue type | `--tissue-type` | Alternative to `--scrna-reference-path`; auto-fetches from CellxGENE Census. |
| Output | `-o` / `--output-path` | `.tsv`, `.csv`, or `.parquet`. Default: `<seg_dir>/validation_metrics.tsv`. |
| Random seed | `--random-seed` | Seed for cell/pair subsampling (default: 0). |

---

## 1. Transcript Assignment Coverage

**Function:** `compute_assignment_metrics` (line 250)
**CLI flag:** `--assigned`
**Direction:** Higher is better (more transcripts assigned)

### Description

Measures the fraction of transcripts successfully assigned to a cell by the segmentation, and counts how many distinct cells and fragment objects were created.

### Formula

$$\text{Assignment\%} = 100 \times \frac{N_{\text{assigned}}}{N_{\text{total}}}$$

Fragment objects are identified by cell IDs with the `fragment-` prefix.

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `transcripts_total` | int | Total transcripts in segmentation file |
| `transcripts_assigned` | int | Transcripts with a valid cell ID |
| `transcripts_assigned_pct` | float | Percentage assigned |
| `cells_assigned` | int | Distinct non-fragment cell IDs |
| `fragments_assigned` | int | Distinct fragment cell IDs (prefix `fragment-`) |

### Parameters

No tunable parameters beyond the input data.

---

## 2. Positive Marker Recall

**Function:** `compute_positive_marker_recall_fast` (line 971)
**CLI flag:** `--positive-marker-recall`
**Direction:** Higher is better
**Requires:** `--scrna-reference-path` or `--tissue-type`

### Description

Measures whether cells express the marker genes expected for their inferred cell type. Cell types are inferred by correlating each segmented cell's gene expression profile against mean cell-type profiles from a scRNA-seq reference.

### Algorithm

1. Build a sparse cell-by-gene count matrix from assigned transcripts.
2. Load scRNA reference and compute mean expression profiles per cell type on shared genes.
3. Assign each cell a host type by maximum cosine similarity to reference profiles.
4. For each cell type, discover up to `n_markers_per_type` (default 12) marker genes where the type's mean expression exceeds the next-highest type by `min_specificity_ratio` (default 1.5x).
5. For each cell, compute recall as the reference-weighted fraction of its marker genes that are present (count > 0):

$$\text{Recall}_i = 100 \times \frac{\sum_{g \in M_t} w_g \cdot \mathbb{1}[x_{ig} > 0]}{\sum_{g \in M_t} w_g}$$

where $M_t$ is the marker gene set for cell $i$'s inferred type $t$, and $w_g$ is the reference expression weight.

6. Final score is the transcript-count-weighted mean across cells.

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `positive_marker_recall_fast` | float | Weighted mean recall (%) |
| `positive_marker_types_used_fast` | int | Cell types with discoverable markers |
| `positive_marker_genes_used_fast` | int | Unique marker genes used |
| `positive_marker_cells_used_fast` | int | Cells scored |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_markers_per_type` | 12 | Max marker genes per cell type |
| `min_specificity_ratio` | 1.5 | Required fold-change over second-highest type |
| `min_transcripts_per_cell` | 20 | Cell filter |
| `max_cells` | 3000 | Subsampling cap |

---

## 3. MECR (Mutually Exclusive Co-expression Rate)

**Function:** `compute_mecr_fast` (line 2287)
**CLI flag:** `--mecr`
**Direction:** Lower is better
**Requires:** `--anndata-path` and either `--me-gene-pairs-path` or `--scrna-reference-path`

### Description

Quantifies how often pairs of genes that should be mutually exclusive (expressed in different cell types) are co-expressed within the same cell. Over-segmentation (merging two cells) inflates MECR.

### Formula

ME gene pairs are loaded from a file or discovered from scRNA-seq. For each pair $(g_1, g_2)$:

**Soft MECR** (default, `soft=True`):

$$\text{MECR}(g_1, g_2) = \frac{\sum_c \min(x_{c,g_1},\; x_{c,g_2})}{\sum_c \max(x_{c,g_1},\; x_{c,g_2})}$$

**Hard MECR** (`soft=False`):

$$\text{MECR}(g_1, g_2) = \frac{|\{c : x_{c,g_1} > 0 \;\land\; x_{c,g_2} > 0\}|}{|\{c : x_{c,g_1} > 0 \;\lor\; x_{c,g_2} > 0\}|}$$

The overall MECR is the unweighted mean across all scored pairs.

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `mecr_fast` | float | Mean MECR across pairs |
| `mecr_pairs_used` | int | Number of gene pairs scored |

### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--anndata-path` / `-a` | None | Path to `segger_segmentation.h5ad` |
| `--me-gene-pairs-path` | None | TSV/CSV with two gene columns |
| `--max-me-gene-pairs` | 500 | Max pairs subsampled |

---

## 4. RESOLVI Contamination

**Function:** `compute_resolvi_contamination_fast` (line 818)
**CLI flag:** `--resolvi`
**Direction:** Lower is better
**Requires:** `--scrna-reference-path` or `--tissue-type`

### Description

Approximates the RESOLVI neighborhood contamination model. For each cell, it estimates what fraction of observed transcripts are better explained by neighboring cells or background than by the cell's own type.

### Algorithm

1. Infer host cell type from scRNA reference (same as Marker Recall).
2. Build a KD-tree over cell centroids and find `k_neighbors` (default 10) nearest neighbors within `max_neighbor_distance` (default 20 $\mu$m).
3. For each cell $i$ with host type $h$, compute expected gene expression as a mixture:

$$\mathbf{p}_i = \alpha_{\text{self}} \cdot \mathbf{r}_h + \alpha_{\text{neighbor}} \cdot \mathbf{p}_{\text{neigh}} + \alpha_{\text{background}} \cdot \mathbf{p}_{\text{bg}}$$

where:
- $\mathbf{r}_h$ = reference profile for host type $h$
- $\mathbf{p}_{\text{neigh}} = \sum_{j \in N(i),\; j \ne h} f_j \cdot \mathbf{r}_j$ (neighbor type frequency-weighted profiles, excluding self type)
- $\mathbf{p}_{\text{bg}} = \sum_t w_t \cdot \mathbf{r}_t$ (global background from cell-count-weighted type frequencies)

4. Compute the self-attribution probability per gene:

$$q_{\text{self}}^{(g)} = \frac{\alpha_{\text{self}} \cdot r_{h,g}}{p_{i,g} + \epsilon}$$

5. A transcript is flagged as contaminated if $q_{\text{self}}^{(g)} < \text{contam\_cutoff}$ (default 0.5).

6. Per-cell contamination % = fraction of transcripts flagged. Overall = transcript-count-weighted mean.

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `resolvi_contamination_pct_fast` | float | Weighted mean contamination (%) |
| `resolvi_contaminated_cells_pct_fast` | float | % cells with any contamination |
| `resolvi_metric_cells_used` | int | Cells scored |
| `resolvi_shared_genes_used` | int | Genes shared with reference |
| `resolvi_cell_types_used` | int | Cell types in reference |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_neighbors` | 10 | Neighbors in KD-tree query |
| `max_neighbor_distance` | 20.0 | Max distance ($\mu$m) for neighbors |
| `alpha_self` | 0.8 | Self-expression mixing weight |
| `alpha_neighbor` | 0.175 | Neighbor mixing weight |
| `alpha_background` | 0.025 | Background mixing weight |
| `contam_cutoff` | 0.5 | $q_{\text{self}}$ threshold |
| `min_transcripts_per_cell` | 20 | Cell filter |
| `max_cells` | 3000 | Subsampling cap |

---

## 5. Spurious Coexpression

**Function:** `compute_spurious_coexpression_fast` (line 1057)
**CLI flag:** `--spurious`
**Direction:** Lower is better
**Requires:** `--source-path` (with `cell_id` and `cell_compartment` columns)

### Description

Detects gene pairs whose co-expression in segmented cells is suspiciously high relative to their nuclear co-occurrence, suggesting the segmentation is merging transcripts from different cells.

### Algorithm

1. **Spatial co-occurrence:** Subsample source transcripts, build a KD-tree (radius default 10 $\mu$m), compute binary Jaccard similarity between gene pairs based on spatial proximity.

2. **Nuclear baseline:** From nuclear transcripts (compartment == 2) in source data, build a cell-by-gene binary matrix and compute Jaccard similarity.

3. **Discover spurious pairs:** Select gene pairs where:
   - `spatial_score / nuclear_score > ratio_cutoff` (default 2.0)
   - `nuclear_score < nuclear_max` (default 0.001)
   - `spatial_score > min_spatial_score` (default 0.001)
   - Both genes have sufficient support (`min_support` default 20)

4. **Score in segmented cells:** For each discovered pair $(g_i, g_j)$, compute Jaccard in the segmented cell-by-gene matrix:

$$J_{\text{seg}}(g_i, g_j) = \frac{\sum_c \min(x_{c,g_i}, x_{c,g_j})}{\sum_c \max(x_{c,g_i}, x_{c,g_j})}$$

5. **Excess co-expression:** $\text{excess} = \max(J_{\text{seg}} - J_{\text{nuc}}, 0)$, weighted by $\log(1 + \max(\text{occur}_{g_i}, \text{occur}_{g_j}))$.

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `spurious_coexpression_fast` | float | Weighted mean excess Jaccard |
| `spurious_pairs_used_fast` | int | Pairs scored in segmented data |
| `spurious_pairs_discovered_fast` | int | Pairs passing discovery filters |
| `spurious_source_transcripts_used_fast` | int | Source transcripts in spatial analysis |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spatial_radius` | 10.0 | KD-tree radius ($\mu$m) |
| `max_spatial_transcripts` | 10,000 | Subsampling cap for spatial step |
| `min_gene_count` | 20 | Min occurrences per gene |
| `min_support` | 20 | Min co-occurrence support |
| `ratio_cutoff` | 2.0 | Spatial/nuclear ratio threshold |
| `nuclear_max` | 0.001 | Max nuclear Jaccard |
| `min_spatial_score` | 0.001 | Min spatial Jaccard |
| `max_pairs` | 200 | Top pairs retained |
| `min_transcripts_per_cell` | 20 | Cell filter |
| `max_cells` | 3000 | Subsampling cap |

---

## 6. Center-Border Similarity (formerly Center-Border NCV)

**Function:** `compute_center_border_ncv_fast` (line 1225)
**CLI flag:** `--center-border-ncv`
**Direction:** Higher is better

### Description

Tests whether a cell's border region has a gene expression profile more similar to its own center than to its neighbors. Well-segmented cells should have internally consistent expression; poorly segmented cells leak neighbor signal at their borders.

### Algorithm

1. For each cell, define center and border zones using an eroded bounding box (erosion = `erosion_fraction` x min side, default 0.3).

2. Build gene expression vectors for center, border, and averaged neighbors (via KD-tree, `n_neighbors` default 10).

3. Compute cosine similarities:
   - $\text{sim}_{\text{center-border}} = \cos(\mathbf{v}_{\text{center}}, \mathbf{v}_{\text{border}})$
   - $\text{sim}_{\text{border-neighbor}} = \cos(\mathbf{v}_{\text{border}}, \mathbf{v}_{\text{neighbor}})$

4. Per-cell score:

$$\text{ratio} = \frac{\text{sim}_{\text{border-neighbor}}}{\text{sim}_{\text{center-border}}}$$

$$\text{score} = \frac{1}{1 + \max(0, \text{ratio} - 1)}$$

A score of 1.0 means the border is more similar to the center than to neighbors. Lower scores indicate border contamination.

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `center_border_ncv_score_fast` | float | Weighted mean score [0, 1] |
| `center_border_ncv_ratio_fast` | float | Weighted mean raw ratio |
| `center_border_ncv_cells_used_fast` | int | Cells scored |

### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--border-erosion-fraction` | 0.3 | Fraction defining center vs border |
| `--border-min-transcripts-per-cell` | 20 | Min transcripts per cell |
| `--border-max-cells` | 3000 | Subsampling cap |

---

## 7. Border Contamination

**Function:** `compute_border_contamination_fast` (line 1520)
**CLI flag:** `--border-contamination`
**Direction:** Lower is better

### Description

Measures whether a cell's border region has disproportionately high transcript density compared to its center, which can indicate that transcripts from neighboring cells are leaking into the periphery.

### Algorithm

1. Define center and border regions using the same eroded bounding box as Center-Border Similarity.

2. Compute area-normalized densities:

$$\rho_{\text{center}} = \frac{N_{\text{center}}}{A_{\text{center}}} \qquad \rho_{\text{border}} = \frac{N_{\text{border}}}{A_{\text{border}}}$$

3. Border enrichment:

$$E = \frac{\rho_{\text{border}}}{\rho_{\text{center}}}$$

4. Contamination score:

$$\text{contam} = \max(E - 1, 0)$$

5. A cell is flagged as contaminated if $E > \text{threshold}$ (default 1.25).

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `border_contamination_fast` | float | Weighted mean contamination score |
| `border_enrichment_fast` | float | Weighted mean enrichment ratio |
| `border_excess_pct_fast` | float | $(E - 1) \times 100$ |
| `border_contaminated_cells_pct_fast` | float | % cells exceeding threshold |
| `border_metric_cells_used` | int | Cells scored |

### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--border-erosion-fraction` | 0.3 | Fraction defining center vs border |
| `--border-min-transcripts-per-cell` | 20 | Min transcripts per cell |
| `--border-max-cells` | 3000 | Subsampling cap |
| `--border-contaminated-enrichment-threshold` | 1.25 | Per-cell enrichment cutoff |

---

## 8. Reference Morphology Match

**Function:** `compute_reference_morphology_match_fast` (line 1419)
**CLI flag:** `--reference-morphology`
**Direction:** Higher is better
**Requires:** `--source-path` (with `cell_id` column in both source and segmentation)

### Description

Compares the morphological properties (area, elongation, circularity) of segmented cells against reference cells from the source platform. Each segmented cell is matched to its best-overlapping reference cell.

### Algorithm

1. Compute bounding-box geometry for both segmented and reference cells:
   - **Area:** $A = \text{width} \times \text{height}$
   - **Elongation:** $E = \max(w, h) / \min(w, h)$
   - **Circularity:** $\Gamma = 4\pi A / P^2$ where $P = 2(w + h)$

2. Match each segmented cell to the reference cell with maximum transcript overlap.

3. Per-cell morphology similarity (average of three components):

$$S_{\text{area}} = \exp\left(-\left|\ln\frac{A_{\text{pred}}}{A_{\text{ref}}}\right|\right)$$

$$S_{\text{elong}} = \exp\left(-\left|\ln\frac{E_{\text{pred}}}{E_{\text{ref}}}\right|\right)$$

$$S_{\text{circ}} = \exp\left(-|\Gamma_{\text{pred}} - \Gamma_{\text{ref}}|\right)$$

$$\text{score} = \frac{S_{\text{area}} + S_{\text{elong}} + S_{\text{circ}}}{3}$$

Weighted by overlap count.

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `reference_morphology_match_fast` | float | Weighted mean score [0, 1] |
| `reference_morphology_cells_used_fast` | int | Matched cell pairs scored |

### Parameters

No additional tunable CLI parameters. Uses default geometry from bounding boxes.

---

## 9. Transcript-Centroid Offset (formerly TCO)

**Function:** `compute_transcript_centroid_offset_fast` (line 1709)
**CLI flag:** `--tco`
**Direction:** Higher is better

### Description

Measures how well-centered a cell's transcript cloud is relative to its bounding box center. Cells where transcripts are skewed to one side may indicate poor boundary placement.

### Formula

1. Transcript centroid: $(\bar{x}, \bar{y})$ = mean of transcript coordinates.
2. Bounding box center: $(c_x, c_y)$ = midpoint of min/max coordinates.
3. Offset distance:

$$d = \sqrt{(\bar{x} - c_x)^2 + (\bar{y} - c_y)^2}$$

4. Score (normalized by cell size):

$$\text{Transcript-Centroid Offset} = \text{clip}\left(1 - \frac{d}{\sqrt{A} + \epsilon},\; 0,\; 1\right)$$

where $A = \text{width} \times \text{height}$.

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `transcript_centroid_offset_fast` | float | Weighted mean transcript-centroid offset score [0, 1] |
| `tco_metric_cells_used` | int | Cells scored |

### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--tco-min-transcripts-per-cell` | 20 | Min transcripts per cell |
| `--tco-max-cells` | 3000 | Subsampling cap |

---

## 10. Signal Doublet (Z-spread)

**Function:** `compute_signal_doublet_fast` (line 1796)
**CLI flag:** `--signal-doublet`
**Direction:** Lower is better (fewer doublet-like cells)
**Requires:** `z` column in transcript data

### Description

Identifies cells whose z-axis spread is abnormally wide, suggesting they may span two vertically stacked cells (doublets).

### Algorithm

1. Compute per-cell z-coordinate standard deviation.
2. Define expected z-spread as the median $\sigma_z$ across all cells with $\sigma_z > 0$.
3. Compute per-cell integrity:

$$I_i = \text{clip}\left(\frac{\tilde{\sigma}_z}{\sigma_{z,i} + \epsilon},\; 0,\; 1\right)$$

4. Flag cell as doublet-like if $I_i < \text{threshold}$ (default 0.6).
5. Report the transcript-count-weighted fraction of flagged cells.

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `signal_doublet_like_fraction_fast` | float | Weighted doublet-like fraction |
| `signal_metric_cells_used` | int | Cells scored |

### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--signal-min-transcripts-per-cell` | 20 | Min transcripts per cell |
| `--signal-max-cells` | 3000 | Subsampling cap |
| `--signal-doublet-threshold` | 0.6 | Integrity threshold for doublet flag |

---

## 11. Vertical Doublet (formerly Signal Hotspot Doublet / VSI)

**Function:** `compute_signal_hotspot_doublet_fast` (line 1914)
**CLI flags:** `--signal-hotspot` or `--vsi`
**Direction:** Lower is better (fewer doublet-like cells)
**Requires:** `--source-path` with z-coordinates

### Description

A more targeted doublet metric that restricts scoring to spatial "hotspot" pixels where vertical signal integrity is already low. This focuses the doublet estimate on regions where the tissue is thick or layered.

### Algorithm

1. **Pixel binning:** Bin source transcripts into $(x, y)$ grid pixels of size `grid_size` (default 3.0 $\mu$m).

2. **Plane-pair cosine similarity:** For each pixel, compute gene-composition cosine similarity between adjacent z-planes:

$$\text{cosine}(z, z+1) = \frac{\mathbf{v}_z \cdot \mathbf{v}_{z+1}}{\|\mathbf{v}_z\| \|\mathbf{v}_{z+1}\|}$$

where $\mathbf{v}_z$ is the gene count vector at plane $z$.

3. **Per-pixel integrity:** Weighted average of plane-pair cosines (weighted by transcript count per pair). Pixels with < `min_pixel_signal` (default 3) transcripts are excluded.

4. **Hotspot detection:** Apply a knee-point algorithm on sorted pixel integrity values to find a low-integrity cutoff. Pixels below this cutoff are "hotspots."

5. **Per-cell scoring:** For cells overlapping hotspot pixels:
   - Split transcripts at the median z-plane.
   - Compute cosine similarity of gene expression between upper and lower halves (cells with < `min_side_transcripts` default 5 on the minor side get coherence = 1.0).
   - Flag as doublet-like if coherence < `doublet_threshold` (default 0.6).

6. Report weighted fraction of flagged cells (weighted by hotspot pixel count).

### Output Keys

| Key | Type | Description |
|-----|------|-------------|
| `signal_hotspot_doublet_fraction_fast` | float | Weighted doublet-like fraction |
| `signal_hotspot_cutoff_fast` | float | Auto-detected integrity cutoff |
| `signal_hotspot_pixels_used_fast` | int | Hotspot pixels |
| `signal_hotspot_candidate_cells_fast` | int | Cells overlapping hotspots |
| `signal_hotspot_metric_cells_used_fast` | int | Cells meeting transcript minimum |
| `signal_hotspot_cells_scored_fast` | int | Cells with enough transcripts on both sides |

### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--signal-hotspot-grid-size` | 3.0 | Pixel bin size ($\mu$m) |
| `--signal-min-transcripts-per-cell` | 20 | Min transcripts per cell |
| `--signal-max-cells` | 3000 | Subsampling cap |
| `--signal-doublet-threshold` | 0.6 | Coherence threshold for doublet flag |

---

## Quick Reference Table

| # | Metric | Direction | Requires scRNA | Requires Source | Requires Z | CLI Flag |
|---|--------|-----------|----------------|-----------------|------------|----------|
| 1 | Transcript Assignment Coverage | Higher = better | No | No | No | `--assigned` |
| 2 | Positive Marker Recall | Higher = better | Yes | No | No | `--positive-marker-recall` |
| 3 | MECR | Lower = better | Yes* | No | No | `--mecr` |
| 4 | RESOLVI Contamination | Lower = better | Yes | No | No | `--resolvi` |
| 5 | Spurious Coexpression | Lower = better | No | Yes** | No | `--spurious` |
| 6 | Center-Border Similarity | Higher = better | No | No | No | `--center-border-ncv` |
| 7 | Border Contamination | Lower = better | No | No | No | `--border-contamination` |
| 8 | Reference Morphology Match | Higher = better | No | Yes | No | `--reference-morphology` |
| 9 | Transcript-Centroid Offset | Higher = better | No | No | No | `--tco` |
| 10 | Signal Doublet (Z-spread) | Lower = better | No | No | Yes | `--signal-doublet` |
| 11 | Vertical Doublet | Lower = better | No | Yes | Yes | `--signal-hotspot` / `--vsi` |

\* MECR requires `--anndata-path` and either `--me-gene-pairs-path` or `--scrna-reference-path` for pair discovery.
\*\* Spurious Coexpression requires source data with `cell_id` and `cell_compartment` columns (nuclear compartment).
