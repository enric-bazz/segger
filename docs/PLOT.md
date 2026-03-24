# Segger Plot Command Reference

Visualize loss curves from training metrics logged by PyTorch Lightning's CSVLogger.

```bash
segger plot -o /path/to/output
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output-directory` | (required) | Segger output directory containing `lightning_logs/.../metrics.csv` |
| `-v` / `--log-version` | None | Lightning log version (e.g., `3` for `version_3`). Default: latest |
| `--quick` | False | Plot in terminal using uniplot (no image saved) |

## Plot Modes

### Standard (PNG)

Renders 2x2 subplot pages using matplotlib. Saves `loss_curves.png` (and `loss_curves_2.png`, etc. for additional pages) to the output directory.

Requires: `pip install segger[plot]` (matplotlib)

### Quick (terminal)

Renders loss curves directly in the terminal using uniplot. No files saved.

Requires: `pip install segger[plot]` (uniplot)

## Metric Discovery

1. Reads `metrics.csv` from Lightning logs
2. Discovers all numeric columns (excludes `epoch` and `step`)
3. Groups columns by base name (splits on `:` prefix for train/val variants)
4. Validation series (`val:*`) render as dashed lines

### Metrics File Resolution

The command searches for `metrics.csv` in this order:

1. `<output_dir>/metrics.csv` (direct)
2. `<output_dir>/lightning_logs/version_<N>/metrics.csv` (specified or latest version)
3. Recursive glob for most recently modified `metrics.csv`

## Interpreting Loss Curves

The logged metrics correspond to the loss components described in [SEGMENT.md](SEGMENT.md#loss-system):

| Curve | What it measures | What to look for |
|-------|-----------------|------------------|
| `loss_tx` | Gene embedding triplet loss | Should decrease steadily. Plateaus early since gene structure is learned quickly |
| `loss_bd` | Cell embedding metric loss | Should decrease. High values indicate cell clusters are poorly separated |
| `loss_sg` | Segmentation assignment loss | Starts appearing when `sg_weight > 0` (default: ramps in over training). Should decrease |
| `loss_align` | ME-gene alignment loss | Only present with `--alignment-loss`. Should decrease slowly; spikes are normal early on |
| `loss` | Weighted total loss | Combined objective. Should generally decrease; may bump when `sg_weight` ramps up |

**Train vs val splits**: Curves prefixed with `train:` and `val:` show training and validation respectively. A widening gap (train decreasing, val flat/increasing) indicates overfitting — try fewer epochs or more data.

**Diagnosing common issues**:
- `loss_sg` not decreasing: The segmentation weight may be too low, or the model hasn't learned good representations yet. Try increasing `--segmentation-loss-weight-end`.
- `loss_tx` or `loss_bd` increasing mid-training: The segmentation loss is dominating. Reduce `--segmentation-loss-weight-end` or increase `--n-epochs` to give representation losses more time.
- `loss_align` staying flat: ME-gene edges may not be present in the data. Check that `--scrna-reference-path` or `--alignment-me-gene-pairs-path` was correctly specified during training.

## Training Duration Context

The number of steps in `metrics.csv` depends on dataset size and tiling parameters. Typical training durations at 20 epochs:

| Dataset Size | Training Time | Approximate Steps |
|:---:|:---:|:---:|
| ~1M transcripts | ~2 min | ~100–200 |
| ~28M transcripts | ~24 min | ~2,000–4,000 |
| ~93M transcripts | ~80 min | ~8,000–15,000 |
| ~125M transcripts | ~125 min | ~10,000–20,000 |

If loss curves show the model hasn't converged (still decreasing at the last step), increase `--n-epochs`. If the model converges in the first quarter, you can save time by reducing epochs.

## Smoothing

All curves are smoothed with a rolling mean:

$$\text{window} = \max\!\left(5,\; \min\!\left(25,\; \lfloor n / 20 \rfloor\right)\right)$$

where $n$ is the number of data points. No smoothing is applied when $n < 3$.

---

## Examples

```bash
# Standard PNG plot (latest version)
segger plot -o output/

# Specific log version
segger plot -o output/ --log-version 3

# Quick terminal plot
segger plot -o output/ --quick
```
