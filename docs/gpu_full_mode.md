# GPU and Chronos-2 Full Mode

The canonical smoke run trains a small PyTorch quantile network on CPU or CUDA when available. It records the actual device, PyTorch version, CUDA device name, epoch losses, and Chronos package availability in `hardware_profile.json`.

For Chronos-2:

```bash
pip install -e .[chronos]
python scripts/chronos2_full_mode.py
support-capacity run --config configs/full.yaml
```

The official Chronos-2 model is optional because model weights may require network access and substantial memory. The adapter constructs a dense 96-step historical context (`lag_96` through `lag_1`, oldest to newest) for every series instead of passing a single lag value. The full-mode adapter fails explicitly rather than silently substituting another model. A model is promoted only if frozen predictive and downstream operational metrics justify the additional compute.
