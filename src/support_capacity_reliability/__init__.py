"""Support capacity reliability platform."""

from __future__ import annotations

import os

# Conservative defaults avoid OpenMP/BLAS oversubscription and intermittent deadlocks when
# tests, LightGBM, SciPy, PyTorch, and simulation run sequentially in one CI job. Users can
# override any value before importing the package for larger full-mode workloads.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

__version__ = "1.4.1rc2"
