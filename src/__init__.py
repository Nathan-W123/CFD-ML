"""CFD-pretrained physics expert for smoke/fog discrimination.

Importing any `src.*` module first runs this file, which pins every numeric
library to one thread (overriding any value already set in the shell). The variables must be set before numpy, numba, torch
or OpenCV are imported, so scripts import `src` (or a submodule) before
anything else. `src.utils.runtime.configure_single_thread` repeats the
runtime calls that environment variables cannot cover (torch, OpenCV).
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_var] = "1"
