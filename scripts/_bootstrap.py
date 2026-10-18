"""Make the repository root importable and pin every library to one thread.

Every script imports this module first (`import _bootstrap  # noqa: F401`), so
`import src` runs before numpy/torch/OpenCV and the thread limits take effect.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: E402,F401  (sets the thread environment variables)
from src.utils.runtime import configure_single_thread  # noqa: E402

THREADS = configure_single_thread()
