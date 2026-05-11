"""Environment-driven configuration shared by the paper code.

All paths are resolved at import time so scripts can be invoked from any
working directory.

Environment variables
---------------------
``LLM_STANCE_CACHE_DIR``
    Where to download/cache HuggingFace model weights and tokenizers. The
    same directory is reused for all four base LLMs. Defaults to the path
    used during the experiments (``/storage3-ciber/parush``); override this
    when running outside the lab cluster.

``LLM_STANCE_LORA_DIR``
    Where the per-target/per-domain LoRA adapters trained by
    ``scripts/train_lora.py`` live (one subfolder per ``model/target``).
    Defaults to ``$LLM_STANCE_CACHE_DIR/lora``.

``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``)
    Optional. If set, ``src.llm_client`` will call ``huggingface_hub.login``
    so gated checkpoints (e.g. Llama-3-8B-Instruct) can be downloaded. We
    deliberately do **not** ship a token in source.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

HF_CACHE_DIR: str = os.environ.get(
    "LLM_STANCE_CACHE_DIR",
    "/storage3-ciber/parush",
)

LORA_DIR: str = os.environ.get(
    "LLM_STANCE_LORA_DIR",
    os.path.join(HF_CACHE_DIR, "lora"),
)

DATASET_PATH: Path = REPO_ROOT / "dataset" / "all_combined.csv"
RESULTS_DIR: Path = REPO_ROOT / "results"
