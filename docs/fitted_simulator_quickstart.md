# Fitted simulator quickstart

Run `ehrdyn-icu fitted-simulator --mode train` with a caller-owned restricted
root. The code trains candidate seeds on `source_model_train`, calibrates only
on `source_model_calibration`, selects against `validation`, and leaves
checkpoints and profiles in the restricted root. Run `--mode evaluate` only
against that same local restricted root.
