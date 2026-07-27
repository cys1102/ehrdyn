# Recursive Controlled-Environment World-Model Entrant

EHRDyn exposes a typed subprocess API for point, independent-
Gaussian, and Gaussian-ensemble recursive transition models. Entrants receive
only observations, masks, recency, previous actions, proposed action sequences,
and public task metadata.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

ehrdyn-icu evaluate-world-model-smoke \
  --entrant world_model_entrant_example/point.json \
  --entrant world_model_entrant_example/gaussian.json \
  --entrant world_model_entrant_example/ensemble.json \
  --output build/world-model-smoke --episodes 8
```

## Full 40-environment validation

```bash
ehrdyn-icu evaluate-world-model-full \
  --entrant recursive_world_model_entrant/entrant.json \
  --output build/world-model-full \
  --forecast-episodes 32 \
  --direct-episodes 512 \
  --ope-datasets 64 \
  --ope-episodes 256 \
  --workers 4
```

The full run emits 40 checkpoint receipts, 440 common-origin recursive horizon
rows, 40 full-episode direct-return rows, and 240 environment-estimator OPE
summary rows. The primary OPE contract refits the cross-fitted denominator and
nuisance dynamics within each of 64 independent 256-episode datasets.

## Metrics and planning

- RMSE and MAE are computed on observed target cells.
- Gaussian NLL and CRPS use the entrant-emitted predictive scale.
- Cov50/80/90/95 and corresponding widths use central Gaussian intervals.
- MACE is the mean absolute coverage error over the four frozen nominal levels.
- Risk--coverage area uses the frozen 0.1 through 1.0 retention grid.
- `H4` means planning lookahead only. Every selected first action is evaluated
  over the full simulator episode.
- H4 uses 64 categorical action sequences, three iterations, eight elites,
  smoothing 0.2, support masking, and first-action execution.
- OPE reports IS, WIS, CWPDIS, DR, WDR, and FQE. Extreme but finite estimates
  are retained rather than filtered.

All 40 mechanisms are constructed public development assets. This workflow is
not an EHR reconstruction, treatment-effect analysis, or clinical policy
evaluation.
