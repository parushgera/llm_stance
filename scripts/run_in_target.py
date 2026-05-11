"""Run base / LoRA in-target stance detection across all six prompts."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

# Make the repo's `src/` package importable regardless of CWD.
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
    create_cot_prompt,
    create_few_shot_prompt,
    create_knowledge_infused_prompt,
    create_vanilla_prompt,
)
from src.utils import get_current_stance_labels, get_few_shot_examples

torch.set_float32_matmul_precision("high")
print(torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

torch._dynamo.config.suppress_errors = True

def run_in_target_experiment(
    df: pd.DataFrame,
    llm_client: Any,
    prompt_type: str,
    target_filter: Optional[str] = None,
    dataset_filter_for_test: Optional[str] = None,
    type_filter: Optional[str] = None,
    knowledge_base: Dict[str, str] = {},
    few_shot_examples: List[Dict[str, str]] = [],
    output_dir: str = str(REPO_ROOT / "results" / "phase1_results" / "in_target"),
    model_name_for_saving: str = "unknown_model",
    save_path: Optional[str] = None,   # <— new
) -> pd.DataFrame:
    
    """
    Runs an in-target stance detection experiment with a specified prompting strategy.
    Filters the DataFrame based on provided criteria.
    """
    filtered_df = df.copy()

    global TARGETS_MAP
    global KNOWLEDGE_BASE
    
    current_stance_labels = get_current_stance_labels(dataset_filter_for_test)

    if target_filter:
        target_full_name = TARGETS_MAP.get(target_filter, target_filter)
        filtered_df = filtered_df[filtered_df['target'] == target_full_name]

    if dataset_filter_for_test:
        filtered_df = filtered_df[filtered_df['dataset'] == dataset_filter_for_test]
    if type_filter:
        filtered_df = filtered_df[filtered_df['type'] == type_filter]

    if filtered_df.empty:
        print(f"No data found for the given filters: Target={target_filter}, Dataset={dataset_filter_for_test}, Type={type_filter}")
        return pd.DataFrame()

    results = []
    for index, row in filtered_df.iterrows():
        tweet_text = row['text']
        true_stance = row['stance']
        predicted_stance = "ERROR"

        # CHANGE 2: Get user content (not full prompt)
        user_content = ""
        
        if prompt_type == "vanilla":
            user_content = create_vanilla_prompt(tweet_text, current_stance_labels)
        elif prompt_type == "knowledge_infused":
            current_knowledge = KNOWLEDGE_BASE.get(target_filter, "")
            if not current_knowledge:
                print(f"Warning: No knowledge found for target '{target_filter}'. Knowledge-infused prompt may be less effective.")
            user_content = create_knowledge_infused_prompt(tweet_text, current_knowledge, current_stance_labels)
        elif prompt_type == "cot":
            user_content = create_cot_prompt(tweet_text, current_stance_labels)
        elif prompt_type == "few_shot":
            user_content = create_few_shot_prompt(tweet_text, few_shot_examples, current_stance_labels)
        elif prompt_type == "cot_knowledge":
            current_knowledge = KNOWLEDGE_BASE.get(target_filter, "")
            if not current_knowledge:
                print(f"Warning: No knowledge found for target '{target_filter}'. CoT+Knowledge prompt may be less effective.")
            user_content = create_cot_knowledge_prompt(tweet_text, current_knowledge, current_stance_labels)
        elif prompt_type == "cot_knowledge_few_shot":
            current_knowledge = KNOWLEDGE_BASE.get(target_filter, "")
            if not current_knowledge:
                print(f"Warning: No knowledge found for target '{target_filter}'. CoT+Knowledge+Few-Shot prompt may be less effective.")
            user_content = create_cot_knowledge_few_shot_prompt(tweet_text, current_knowledge, few_shot_examples, current_stance_labels)
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
            'target_filter': target_filter,
            'dataset_filter_for_test': dataset_filter_for_test,
            'type_filter': type_filter,
            'prompt': formatted_prompt  # Store formatted prompt
        })

    results_df = pd.DataFrame(results)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        filepath = save_path
    else:
        os.makedirs(output_dir, exist_ok=True)
        filename_parts = [
            model_name_for_saving,
            prompt_type,
            f"target-{target_filter}",
            f"dataset-{dataset_filter_for_test}",
            f"type-{type_filter}",
            "results.csv",
        ]
        filename = "_".join(part.replace(" ", "_").replace("/", "-") for part in filename_parts if part)
        filepath = os.path.join(output_dir, filename)

    print(f"Saving results to: {filepath}")
    results_df.to_csv(filepath, index=False)
    return results_df

# The main execution remains the same
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--lora-dir", type=str, default=LORA_DIR)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "vanilla",
            "knowledge_infused",
            "few_shot",
            "cot",
            "cot_knowledge",
            "cot_knowledge_few_shot",
        ],
        help="Subset of prompt strategies to run.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TARGETS_MAP.keys()),
        help="Target keys to evaluate (defaults to all 13 targets + 2 domains).",
    )
    args = parser.parse_args()

    large_storage_path = HF_CACHE_DIR
    PROMPT_TYPES = args.prompts
    ALL_TARGET_KEYS = args.targets

    df_dataset = pd.read_csv(DATASET_PATH)
    test_dataset_split = "test"
    output_directory = str(
        REPO_ROOT / "results"
        / ("phase3_results" if args.use_lora else "phase1_results")
        / "in_target"
    )
    os.makedirs(output_directory, exist_ok=True)

    for model_key in MODEL_CONFIG:
        print(f"Running experiments for model: {model_key}")

        # load per-target when using LoRA; otherwise once per model
        base_client = None
        if not args.use_lora:
            base_client = LLMClient(model_key, cache_dir=large_storage_path)

        for target_key in ALL_TARGET_KEYS:
            n_examples_per_class = 5

            few_shots = get_few_shot_examples(
                df_dataset,
                n_examples_per_class=n_examples_per_class,
                target_filter=target_key,
                type_filter='train',
                dataset_filter=TARGET_DATASET_MAP.get(target_key, '')
            )
            
            # CHANGE: Add explicit memory cleanup before loading new LoRA
            if args.use_lora:
                # Clear any existing CUDA cache before loading new adapter
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                
                adapter_path = os.path.join(args.lora_dir, model_key, target_key)
                llm_client = LLMClient(model_key, cache_dir=large_storage_path, adapter_path=adapter_path)
            else:
                llm_client = base_client

            for p_type in PROMPT_TYPES:
                print(f"Running Prompt Type: {p_type} on target/domain key {target_key}: {TARGETS_MAP.get(target_key)} ")
                dataset_for_test = TARGET_DATASET_MAP.get(target_key, '')
                type_filter_val = 'test'
                filename_parts = [
                    model_key, p_type,
                    f"target-{target_key}",
                    f"dataset-{dataset_for_test}",
                    f"type-{type_filter_val}",
                    "results.csv",
                ]
                filename = "_".join(part.replace(" ", "_").replace("/", "-") for part in filename_parts if part)
                expected_path = os.path.join(output_directory, filename)
                if os.path.exists(expected_path):
                    print(f"⏭️ Skipping (already exists): {expected_path}")
                    continue

                results_df = run_in_target_experiment(
                    df=df_dataset,
                    llm_client=llm_client,
                    prompt_type=p_type,
                    target_filter=target_key,
                    dataset_filter_for_test=dataset_for_test,
                    type_filter=type_filter_val,
                    few_shot_examples=few_shots,
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