"""Run base / LoRA cross-target and cross-domain stance detection."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DATASET_PATH, HF_CACHE_DIR, LORA_DIR
from src.llm_client import LLMClient
from src.mappings import (
    KNOWLEDGE_BASE,
    SEMEVAL_LABELS,
    TARGET_DATASET_MAP,
    TARGETS_MAP,
    WTWT_LABELS,
)
from src.model_config import MODEL_CONFIG
from src.prompts import (
    chat_manager,
    create_cot_knowledge_few_shot_prompt,
    create_cot_knowledge_prompt,
    create_few_shot_prompt,
    create_knowledge_infused_prompt,
)
from src.utils import get_current_stance_labels, get_few_shot_examples

torch.set_float32_matmul_precision("high")
print(torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
torch._dynamo.config.suppress_errors = True

# Reciprocal cross-target pairs (SemEval) + reciprocal cross-domain pair (WT-WT).
# Matches Section 3.1 of the paper.
ALL_CROSS_PAIRS = [
    ("dt", "hc"),
    ("hc", "dt"),
    ("fm", "la"),
    ("la", "fm"),
    ("ent", "hlt"),  # cross-domain (entertainment <-> healthcare, WT-WT)
    ("hlt", "ent"),
]
CROSS_TARGET_PAIRS = ALL_CROSS_PAIRS

def run_cross_target_experiment(
    df: pd.DataFrame,
    llm_client: Any,
    prompt_type: str,
    source_target: str,
    dest_target: str,
    source_dataset: str,
    dest_dataset: str,
    type_filter: Optional[str] = 'test',
    output_dir: str = str(REPO_ROOT / "results" / "phase1_results" / "cross_target"),
    model_name_for_saving: str = "unknown_model",
    save_path: Optional[str] = None,  # NEW
) -> pd.DataFrame:
    
    # Filter destination data for testing
    filtered_dest_df = df[
        (df['target'] == TARGETS_MAP.get(dest_target, dest_target)) &
        (df['dataset'] == dest_dataset)
    ]
    if type_filter:
        filtered_dest_df = filtered_dest_df[filtered_dest_df['type'] == type_filter]
    if filtered_dest_df.empty:
        print(f"No data for dest target={dest_target}, dataset={dest_dataset}, type={type_filter}")
        return pd.DataFrame()

    # Get source knowledge and few-shot examples
    source_knowledge = KNOWLEDGE_BASE.get(source_target, "")
    few_shots = get_few_shot_examples(
        df,
        n_examples_per_class=5,
        target_filter=source_target,
        type_filter='train',
        dataset_filter=source_dataset
    )
    stance_labels = get_current_stance_labels(dest_dataset)

    results = []
    for _, row in filtered_dest_df.iterrows():
        tweet_text = row['text']
        true_stance = row['stance']
        
        # CHANGE 2: Generate user content (not full prompt)
        user_content = ""
        
        if prompt_type == "knowledge_infused":
            if not source_knowledge:
                print(f"Warning: No knowledge found for source target '{source_target}'. Knowledge-infused prompt may be less effective.")
            user_content = create_knowledge_infused_prompt(tweet_text, source_knowledge, stance_labels)
            
        elif prompt_type == "few_shot":
            user_content = create_few_shot_prompt(tweet_text, few_shots, stance_labels)
            
        elif prompt_type == "cot_knowledge":
            if not source_knowledge:
                print(f"Warning: No knowledge found for source target '{source_target}'. CoT+Knowledge prompt may be less effective.")
            user_content = create_cot_knowledge_prompt(tweet_text, source_knowledge, stance_labels)
            
        elif prompt_type == "cot_knowledge_few_shot":
            if not source_knowledge:
                print(f"Warning: No knowledge found for source target '{source_target}'. CoT+Knowledge+Few-Shot prompt may be less effective.")
            user_content = create_cot_knowledge_few_shot_prompt(tweet_text, source_knowledge, few_shots, stance_labels)
            
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # CHANGE 3: Use chat template generation
        predicted_stance = llm_client.generate_text(
            user_content, 
            max_new_tokens=30, 
            temperature=0.1,
            use_chat_template=True  # Enable chat template
        ).strip()
        
        # CHANGE 4: Store the formatted prompt for debugging
        formatted_prompt = chat_manager.format_prompt(llm_client.model_key, user_content)
        #print(formatted_prompt)
        #print(f"Predicted Stance: {predicted_stance}")

        results.append({
            'tweet': tweet_text,
            'true_stance': true_stance,
            'predicted_stance': predicted_stance,
            'prompt_type': prompt_type,
            'source_target': source_target,
            'dest_target': dest_target,
            'source_dataset': source_dataset,
            'dest_dataset': dest_dataset,
            'type_filter': type_filter,
            'prompt': formatted_prompt  # Store formatted prompt
        })

    results_df = pd.DataFrame(results)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        filepath = save_path
    else:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{model_name_for_saving}_{prompt_type}_src-{source_target}_dst-{dest_target}_results.csv"
        filepath = os.path.join(output_dir, filename)

    print(f"Saving results to: {filepath}")
    results_df.to_csv(filepath, index=False)
    return results_df

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--lora-dir", type=str, default=LORA_DIR)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "knowledge_infused",
            "few_shot",
            "cot_knowledge",
            "cot_knowledge_few_shot",
        ],
        help="Subset of the four cross-context prompts to run.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Source-destination pairs as 'src:dst' (e.g. dt:hc fm:la). "
        "Defaults to the six reciprocal pairs from the paper.",
    )
    args = parser.parse_args()

    large_storage_path = HF_CACHE_DIR
    PROMPT_TYPES = args.prompts

    if args.pairs:
        CROSS_TARGET_PAIRS = [tuple(p.split(":", 1)) for p in args.pairs]
    else:
        CROSS_TARGET_PAIRS = ALL_CROSS_PAIRS

    df_dataset = pd.read_csv(DATASET_PATH)
    output_directory = str(
        REPO_ROOT / "results"
        / ("phase3_results" if args.use_lora else "phase1_results")
        / "cross_target"
    )
    os.makedirs(output_directory, exist_ok=True)
    # MODEL_CONFIG = {
    #     "mistral_24b": "mistralai/Mistral-Small-Instruct-2409"
    # }
    for model_key in MODEL_CONFIG:
        print(f"Running cross-target experiments for model: {model_key}")
        base_client = None
        if not args.use_lora:
            base_client = LLMClient(model_key, cache_dir=large_storage_path)

        for (source_target, dest_target) in CROSS_TARGET_PAIRS:
            source_dataset = TARGET_DATASET_MAP.get(source_target, '')
            dest_dataset = TARGET_DATASET_MAP.get(dest_target, '')

            # CHANGE: Add explicit memory cleanup before loading new LoRA
            if args.use_lora:
                # Clear any existing CUDA cache before loading new adapter
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                
                adapter_path = os.path.join(args.lora_dir, model_key, source_target)
                llm_client = LLMClient(model_key, cache_dir=large_storage_path, adapter_path=adapter_path)
            else:
                llm_client = base_client

            for p_type in PROMPT_TYPES:
                filename = f"{model_key}_{p_type}_src-{source_target}_dst-{dest_target}_results.csv"
                expected_path = os.path.join(output_directory, filename)
                if os.path.exists(expected_path):
                    print(f"⏭️ Skipping (already exists): {expected_path}")
                    continue
                print(f"Running cross-target experiment for {model_key} with prompt type {p_type} and source target {source_target} and dest target {dest_target}")
                run_cross_target_experiment(
                    df=df_dataset,
                    llm_client=llm_client,
                    prompt_type=p_type,
                    source_target=source_target,
                    dest_target=dest_target,
                    source_dataset=source_dataset,
                    dest_dataset=dest_dataset,
                    type_filter='test',
                    output_dir=output_directory,
                    model_name_for_saving=model_key,
                    save_path=expected_path,
                ) 
            # CHANGE: More aggressive memory cleanup when using LoRA
            if args.use_lora:
                # Delete client and clear memory more thoroughly
                del llm_client
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force garbage collection
                import gc
                gc.collect() 