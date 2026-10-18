"""Shared pytest setup: one thread everywhere, repo root importable."""

import src  # noqa: F401  (sets the thread environment variables first)
from src.utils.runtime import configure_single_thread

configure_single_thread()
