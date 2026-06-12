"""Redirect torch and HuggingFace model caches to a project-local models/ dir.

Imported from `aivision.__init__` so any subsequent torch.hub / huggingface_hub
call lands under `./models/` rather than `~/.cache/`. Uses `setdefault` so a
user-set env var still wins.
"""

from __future__ import annotations

import os
from pathlib import Path


def models_dir() -> Path:
    """Project-local models directory: <repo_root>/models/."""
    # src/aivision/model_cache.py -> parents[2] is the repo root
    return Path(__file__).resolve().parents[2] / "models"


def configure() -> None:
    d = models_dir()
    (d / "torch").mkdir(parents=True, exist_ok=True)
    (d / "huggingface").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_HOME", str(d / "torch"))
    os.environ.setdefault("HF_HOME", str(d / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(d / "huggingface" / "hub"))
