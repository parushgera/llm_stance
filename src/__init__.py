"""Code for the *SEM 2026 paper "Diagnosing Generalization in Open-Source LLMs
for Stance Detection" (co-located with ACL 2026).

Importable helpers used by the CLI scripts under ``scripts/``:

- :mod:`src.config`        - env-driven paths (HF cache dir, LoRA dir).
- :mod:`src.model_config`  - HuggingFace model IDs for the four open-source LLMs.
- :mod:`src.mappings`      - target/dataset/knowledge-base lookups.
- :mod:`src.prompts`       - chat-template manager + the six prompt builders.
- :mod:`src.llm_client`    - thin wrapper around HF transformers + LoRA adapters.
- :mod:`src.utils`         - few-shot sampler and stance-label helpers.
"""

from .config import HF_CACHE_DIR, LORA_DIR, REPO_ROOT
from .model_config import MODEL_CONFIG
from .mappings import (
    KNOWLEDGE_BASE,
    SEMEVAL_LABELS,
    TARGET_DATASET_MAP,
    TARGET_MODULES_MAP,
    TARGETS_MAP,
    WTWT_LABELS,
)

__all__ = [
    "HF_CACHE_DIR",
    "KNOWLEDGE_BASE",
    "LORA_DIR",
    "MODEL_CONFIG",
    "REPO_ROOT",
    "SEMEVAL_LABELS",
    "TARGETS_MAP",
    "TARGET_DATASET_MAP",
    "TARGET_MODULES_MAP",
    "WTWT_LABELS",
]
