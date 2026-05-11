# Diagnosing Generalization in Open-Source LLMs for Stance Detection

> ⚠️ **Working repository.** The associated paper is accepted but not yet
> published; the BibTeX entry below is a placeholder and will be replaced
> with the official citation once proceedings are released. The paper PDF
> is **not** redistributed in this repository.

This repository contains the code, prompts, LoRA configurations, and analysis
artefacts for the paper:

> **Diagnosing Generalization in Open-Source LLMs for Stance Detection.**
> Parush Gera and Tempestt Neal.
> *Proceedings of the 15th Joint Conference on Lexical and Computational
> Semantics (\*SEM 2026), co-located with ACL 2026.*

The paper is a **diagnostic study** of four open-source instruction-tuned
LLMs (Phi-3-mini-3.8B, Mistral-7B, Llama-3-8B, Mistral-Small-24B), evaluated
across three stance-detection regimes: **in-target (SD)**, **cross-target
(SDCT)**, and **cross-domain (SDCD)**. We isolate how *model size*, *prompt
design*, and *parameter-efficient fine-tuning* (LoRA) interact across **912
controlled experiments**.

Three findings drive the rest of this codebase:

1. **Scale helps prompting, not adaptation.** Larger models prompt better in
   in-target settings, but the advantage shrinks (and sometimes reverses)
   after LoRA fine-tuning.
2. **LoRA specializes at the cost of transfer.** Across all four models,
   LoRA improves in-target macro-F1 by **+48.6%** but degrades
   cross-target/cross-domain macro-F1 by **−25.5%** on average.
3. **Optimal prompts depend on scale.** Small models like enriched CoT +
   knowledge prompts; large models are best with few-shot.

---

## Repository layout

```
.
├── README.md                       # this file
├── requirements.txt                # pinned Python dependencies
├── .gitignore
│
├── src/                            # importable package: paper helpers
│   ├── __init__.py
│   ├── config.py                   # env-driven cache / LoRA paths
│   ├── model_config.py             # HF model IDs (Phi, M7B, L3-8B, M24B)
│   ├── mappings.py                 # target -> dataset / knowledge mappings
│   ├── prompts.py                  # chat-template manager + 6 prompt builders
│   ├── llm_client.py               # HF transformers wrapper, optional LoRA
│   └── utils.py                    # few-shot sampler, label helpers
│
├── scripts/                        # CLI entry points + SLURM launchers
│   ├── run_in_target.py            # SD inference, base or LoRA
│   ├── run_cross_target.py         # SDCT + SDCD inference, base or LoRA
│   ├── train_lora.py               # 60 LoRA adapters (4 models × 15 targets)
│   ├── parse_base.py               # robust label parsing, base outputs
│   ├── parse_lora.py               # robust label parsing, LoRA outputs
│   ├── compute_f1.py               # validates + computes macro-F1
│   ├── make_violin_plots.py        # Figures 1, 2, 3
│   └── *.sh                        # SLURM launchers (set up for our cluster)
│
├── notebooks/
│   └── analysis.ipynb              # tables, deltas, prompt comparisons
│
├── dataset/
│   └── all_combined.csv            # the four merged stance datasets
│
├── analysis_results_for_paper/     # CSV outputs that back the paper tables
│   ├── lora_effectiveness/         # Tables 6, 7  (RQ3)
│   ├── prompt_analysis/            # Table 4      (RQ2)
│   ├── parameter_size_analysis/    # Table 3      (RQ1)
│   └── consistency_analysis/       # supplementary
│
└── results/                        # tracked; produced by the scripts
    ├── phase1_results/             # raw base-LLM outputs
    ├── phase3_results/             # raw LoRA outputs
    ├── parsed_phase1_results/      # after scripts/parse_base.py
    ├── parsed_phase3_results/      # after scripts/parse_lora.py
    ├── f1_scores/                  # combined_f1_scores.csv (used by plots)
    └── plots/violins/              # Figures 1, 2, 3 (.png + .pdf + .csv)
```

`archive/`, `dissertation/`, and `political-bias/` are kept locally for
provenance / follow-up work and are excluded by `.gitignore`.

---

## Datasets

We use four publicly available stance-detection benchmarks, merged into a
single `dataset/all_combined.csv` with consistent `text / target / stance /
dataset / type` columns (`type ∈ {train, test}`):

| Dataset                                              | Targets / Domains                                     | Train / Test  |
| ---------------------------------------------------- | ----------------------------------------------------- | ------------- |
| **SemEval-2016 Task 6** (Mohammad et al., 2016)      | `at, cc, dt, fm, hc, la`                              | 3,444 / 1,426 |
| **P-Stance** (Li et al., 2021)                       | `bernie, dtp, joe`                                    | 19,417 / 2,157 |
| **COVID-19-Stance** (Glandt et al., 2021)            | `face, fauci, school, stay`                           | 5,333 / 800   |
| **WT-WT** (Conforti et al., 2020) — cross-domain     | `ent` (entertainment) / `hlt` (healthcare)            | 33,242 / 11,081 |

Reciprocal cross-target/cross-domain pairs follow prior work and Section 3.1
of the paper:

- **SDCT** (SemEval): `dt ↔ hc`, `fm ↔ la`
- **SDCD** (WT-WT): `ent ↔ hlt`

The original datasets must be obtained from their respective authors. Our
`all_combined.csv` is provided here purely as a working artefact.

---

## Models, prompts, and LoRA

### Models (HF IDs in `src/model_config.py`)

| Key            | HuggingFace ID                              | Size  |
| -------------- | ------------------------------------------- | ----- |
| `phi`          | `microsoft/Phi-3-mini-128k-instruct`        | 3.8B  |
| `mistral_7b`   | `mistralai/Mistral-7B-Instruct-v0.3`        | 7B    |
| `llama3_8b`    | `meta-llama/Meta-Llama-3-8B-Instruct`       | 8B    |
| `mistral_24b`  | `mistralai/Mistral-Small-Instruct-2409`     | 24B   |

Llama-3 is gated; export `HF_TOKEN=...` before running so `src/llm_client.py`
can authenticate at import time.

### Prompts (`src/prompts.py`)

The six strategies from Section 3.4.1 of the paper, each available as a
builder function:

| Key                       | Strategy                                       |
| ------------------------- | ---------------------------------------------- |
| `vanilla`                 | Zero-shot (P\_ZS)                              |
| `few_shot`                | Few-shot, k=5 per stance label (P\_FS)         |
| `knowledge_infused`       | Target/domain knowledge block (P\_KI)          |
| `cot`                     | Chain-of-Thought, label-only output (P\_CoT)   |
| `cot_knowledge`           | CoT + knowledge (P\_CoT+KI)                    |
| `cot_knowledge_few_shot`  | CoT + knowledge + few-shot (P\_CoT+KI+FS)      |

Each builder returns the *user* turn; `chat_manager.format_prompt()` wraps it
with the model-specific chat template. The system instruction in
`SYSTEM_INSTRUCTION` is used for all strategies and includes the research
disclaimer described in Section 3.4.1.

### LoRA (`scripts/train_lora.py`)

A single LoRA configuration is used across all models and datasets, exactly
as reported in Table 2 of the paper:

- `r = 8`, `α = 16`, `bias = "none"`, FP16 precision, warmup ratio `0.1`
- Per-dataset overrides for dropout / batch size / epochs / LR are encoded
  in `get_dynamic_config()` (SemEval+COVID vs P-Stance vs WT-WT).
- Adapter target modules per architecture come from
  `TARGET_MODULES_MAP` in `src/mappings.py`.

This produces 60 adapters (4 models × 15 targets/domains) under
`$LLM_STANCE_LORA_DIR/<model>/<target>/`.

---

## Setup

```bash
git clone https://github.com/parushgera/llm_stance.git
cd llm_stance

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# (optional) where HF weights and LoRA adapters live; both default to
# /storage3-ciber/parush, which only exists on our cluster.
export LLM_STANCE_CACHE_DIR=/path/to/big/cache
export LLM_STANCE_LORA_DIR=$LLM_STANCE_CACHE_DIR/lora

# (required) for gated models (Llama-3-8B-Instruct).
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

A CUDA-matched build of `torch` is needed to run the experiments on GPU. The
24B base + LoRA fine-tuning configurations expect 2× H100 (~80 GB each).

---

## Reproducing the paper

The pipeline has four phases. All scripts can be invoked from any working
directory; they discover the repo root from their own location and read /
write paths under `results/` accordingly.

### 1. Train LoRA adapters (~ once per model × target)

```bash
# All four base models × all 15 targets/domains.
python scripts/train_lora.py

# Subset (recommended for a sanity run):
python scripts/train_lora.py --models phi mistral_7b
```

Adapters are saved under `$LLM_STANCE_LORA_DIR/<model_key>/<target_key>/`,
together with `training_history.csv` and `training_time.txt`.

### 2. Run inference (720 + 128 + 64 = 912 experiments)

```bash
# In-target SD: six prompts × four models × 15 targets/domains.
python scripts/run_in_target.py                        # Base
python scripts/run_in_target.py --use-lora             # LoRA

# Cross-target SDCT + cross-domain SDCD: four prompts × four models × six pairs.
python scripts/run_cross_target.py                     # Base
python scripts/run_cross_target.py --use-lora          # LoRA
```

Both scripts skip CSVs that already exist under `results/phase{1,3}_results/`
so a partially-completed run can simply be re-launched. Use `--prompts`,
`--targets`, or `--pairs` to subset.

### 3. Parse outputs

LLM responses are not always cleanly formatted as `{label: STANCE}`. Both
parsers run a structured-then-fallback regex parse, and re-prompt with an
explicit JSON-output constraint (up to 20 attempts) before falling back to
the dataset-appropriate default label.

```bash
# Base outputs (one model at a time keeps memory bounded).
python scripts/parse_base.py --model phi          --no-clarify
python scripts/parse_base.py --model mistral_7b   --no-clarify
python scripts/parse_base.py --model llama3_8b    --no-clarify
python scripts/parse_base.py --model mistral_24b  --no-clarify

# LoRA outputs (each adapter is loaded per target, then released).
python scripts/parse_lora.py --model phi
python scripts/parse_lora.py --model mistral_7b
python scripts/parse_lora.py --model llama3_8b
python scripts/parse_lora.py --model mistral_24b
```

### 4. Compute metrics + figures

```bash
# Validates labels and writes results/f1_scores/combined_f1_scores.csv
python scripts/compute_f1.py

# Renders Figures 1, 2, 3 and the supplementary violins under
# results/plots/violins/.
python scripts/make_violin_plots.py
```

`notebooks/analysis.ipynb` reads `results/f1_scores/combined_f1_scores.csv`
and emits the per-table CSVs in `analysis_results_for_paper/` (LoRA
effectiveness, prompt analysis, parameter-size analysis, consistency).

### Cluster (SLURM) shortcuts

The `.sh` siblings in `scripts/` wrap every step with our SLURM headers
(partition `SIPEIE23`, GPU48, 200 GB RAM, 1–2× H100). Adjust to your
cluster, then:

```bash
sbatch scripts/train_lora.sh
sbatch scripts/run_in_target_base.sh
sbatch scripts/run_in_target_lora.sh
sbatch scripts/run_cross_target_base.sh
sbatch scripts/run_cross_target_lora.sh
sbatch scripts/parse_phi.sh         # repeat per model
sbatch scripts/parse_mistral_7b.sh
sbatch scripts/parse_llama3_8b.sh
sbatch scripts/parse_mistral_24b.sh
```

---

## Where each paper artefact lives

| Paper element                                   | Source                                                           |
| ----------------------------------------------- | ---------------------------------------------------------------- |
| Table 3 (parameter size analysis)               | `analysis_results_for_paper/parameter_size_analysis/`            |
| Table 4 (prompt analysis)                       | `analysis_results_for_paper/prompt_analysis/`                    |
| Table 5 (best prompt by model size)             | `analysis_results_for_paper/best_prompt_performance_by_model.csv`|
| Tables 6, 7 (LoRA effectiveness)                | `analysis_results_for_paper/lora_effectiveness/`                 |
| Figure 1 (size violins)                         | `results/plots/violins/violin_3_*` + `violin_4_*`                |
| Figure 2 (prompt violins)                       | `results/plots/violins/violin_2_*` + `violin_6_*`                |
| Figure 3 (LoRA gain violins)                    | `results/plots/violins/violin_3_*`                               |
| Listings 1–6 (prompt examples)                  | `src/prompts.py` (one builder per listing)                       |

---

## Citation

A placeholder — to be replaced with the official BibTeX entry once \*SEM 2026
proceedings are published:

```bibtex
@inproceedings{gera2026diagnosing,
  title     = {Diagnosing Generalization in Open-Source LLMs for Stance Detection},
  author    = {Gera, Parush and Neal, Tempestt},
  booktitle = {Proceedings of the 15th Joint Conference on Lexical and
               Computational Semantics (*SEM 2026)},
  note      = {Co-located with ACL 2026},
  year      = {2026}
}
```

## Contact

Questions, issues, or reproduction problems: please open a GitHub issue or
email **parush.edu@gmail.com** (Parush Gera).
