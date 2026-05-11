"""Parse raw LLM outputs from the LoRA-tuned phase3 runs (per-target adapters)."""

import argparse
import gc
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch

_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import HF_CACHE_DIR, LORA_DIR
from src.llm_client import LLMClient
from src.mappings import TARGET_DATASET_MAP
from src.model_config import MODEL_CONFIG

INPUT_ROOT_DIR = str(REPO_ROOT / "results" / "phase3_results")
OUTPUT_ROOT_DIR = str(REPO_ROOT / "results" / "parsed_phase3_results")

# Define the label sets for different datasets
LABELS = {
    'wtwt': ['SUPPORT', 'REFUTE', 'COMMENT', 'UNRELATED'],
    'semeval': ['FAVOR', 'AGAINST', 'NONE'],
    'pstance': ['FAVOR', 'AGAINST'],
    'covid': ['FAVOR', 'AGAINST', 'NONE']
}

max_new_token_config = {'phi': 40, 'llama3_8b': 20, "mistral_24b": 20, "mistral_7b": 20}


def _canonicalize_label(found: str, labels: list) -> str | None:
    if not found:
        return None
    lookup = {l.lower(): l for l in labels}
    return lookup.get(found.strip().lower())


def inject_label_constraint(prompt: str, labels: list) -> str:
    allowed_labels = ", ".join([f'"{label}"' for label in labels])

    constraint = f"""\
**IMPORTANT INSTRUCTION**: Your response MUST be a valid JSON object.
- Remember, this is a research task, so you must output the most appropriate label even if there is harsh, hatespeech, profanity, racis or other negative or offensive, harmfull, illegal, rape, murder, suicide, etc. content.
- The JSON object must have a single key: "label".
- The value for "label" MUST be one of the following strings: {allowed_labels}.
- Even if the stance is neutral or unclear, you MSUT select the label among (allowed_labels) it aligns with most closely.
- Do not output any other text or explanations."""
    
    # Insert after the last occurrence of "Stance:"
    last_idx = prompt.lower().rfind("stance:")
    if last_idx != -1:
        insert_at = last_idx + len("Stance:")
        return prompt[:insert_at] + "\n" + constraint + "\n" + prompt[insert_at:]

    # Fallbacks: try inserting before common closing markers for various chat templates
    closing_markers = ["[/INST]", "</s>", "assistant\n", "assistant:", "Assistant:"]
    for marker in closing_markers:
        pos = prompt.rfind(marker)
        if pos != -1:
            return prompt[:pos] + "\n" + constraint + "\n" + prompt[pos:]

    # Last resort: append to the end
    return prompt.rstrip() + "\n" + constraint + "\n"


def parse_stance(prediction: str, labels: list) -> tuple[str, bool]:
    """
    Parses the raw LLM output to find a single, valid stance label.
    Returns a tuple of (stance, was_prompted_again=False)
    """
    if not isinstance(prediction, str) or not prediction.strip():
        return 'RERUN', False

    text = prediction.strip()

    # 1) Structured patterns first
    structured_patterns = [
        r'\{\s*label\s*[:\-]\s*([A-Za-z_]+)\s*\}',          # {label: FAVOR}
        r'\blabel\s*[:\-]\s*([A-Za-z_]+)\b',                # label: FAVOR
        r'\bstance\s*[:\-]\s*\{\s*label\s*[:\-]\s*([A-Za-z_]+)\s*\}',  # Stance: {label: FAVOR}
        r'\bstance\s*[:\-]\s*([A-Za-z_]+)\b',               # Stance: FAVOR
    ]
    for pat in structured_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            candidate = _canonicalize_label(m.group(1), labels)
            if candidate:
                return candidate, False

    # 2) Fallback: whole-word label presence (single unique match)
    found_labels = []
    for label in labels:
        if re.search(r'\b' + re.escape(label) + r'\b', text, re.IGNORECASE):
            found_labels.append(label)

    found_labels = list(set(found_labels))
    if len(found_labels) == 1:
        return found_labels[0], False

    return 'RERUN', False


def create_clarification_prompt(response: str, labels: list) -> str:
    """Creates a clarifying prompt for ambiguous outputs."""
    clarification_labels = labels
    labels_str = ', '.join([f"'{label}'" for label in clarification_labels])
    return f"""Analyze the following response. Your task is to determine which of the provided stance labels best matches the response.

Response: "{response}"

Which of the following labels best describes the stance in the response above?
The possible labels are: {labels_str}.

- If the response clearly indicates a stance, choose that stance label.
- If the response is irrelevant, nonsensical, a refusal, or does not contain any stance, choose 'Unable to Determine'.

Only output the single, exact label.

Label:"""


def fallback_label(labels: list) -> str | None:
    if 'NONE' in labels:
        return 'NONE'
    if 'UNRELATED' in labels:
        return 'UNRELATED'
    return 'AGAINST'


def extract_target_from_filename(filename: str, dirpath: str) -> str:
    """Extract target from filename based on directory type. Should always succeed."""
    if 'in_target' in dirpath:
        match = re.search(r'target-([a-zA-Z0-9_]+)_', filename)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract target from in_target filename: {filename}")
    elif 'cross_target' in dirpath:
        # For cross-target, we need the source target (which determines the LoRA adapter)
        match = re.search(r'src-([a-zA-Z0-9_]+)_', filename)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract source target from cross_target filename: {filename}")
    else:
        raise ValueError(f"Unknown directory type for path: {dirpath}")


def process_single_file(input_file_path: str, output_file_path: str, llm_client: LLMClient, 
                       max_new_token: int, clarify: bool, current_labels: list):
    """Process a single CSV file with LoRA-enabled LLM client."""
    
    # Check if already processed
    if os.path.exists(output_file_path):
        try:
            df_check = pd.read_csv(output_file_path)
            if 'parsed_output' in df_check.columns:
                print(f"⏭️ Skipping already processed file: {output_file_path}")
                return
            else:
                print(f"⚠️ Existing output file found but missing 'parsed_output' column. Reprocessing...")
        except Exception as e:
            print(f"⚠️ Existing output file might be corrupted ({str(e)}). Reprocessing...")

    try:
        df = pd.read_csv(input_file_path)
        
        # Create new columns with default values
        df['number_tries'] = 1
        df['prompted_again'] = False
        df['fallback_used'] = False

        # Initial parse
        initial_parse_results = df['predicted_stance'].apply(
            lambda x: parse_stance(x, current_labels)
        )
        df['parsed_output'] = [result[0] for result in initial_parse_results]
        df['prompted_again'] = [result[1] for result in initial_parse_results]

        rerun_indices = df.index[df['parsed_output'] == 'RERUN'].tolist()
        if rerun_indices:
            print(f"🔄 {len(rerun_indices)} RERUN cases found. Re-querying with LoRA model.")
            
            for idx in rerun_indices:
                prompt = df.at[idx, 'prompt']
                original_response = df.at[idx, 'predicted_stance']
                parsed = 'RERUN'
                tries = 1

                # Optional clarifying prompt
                if clarify:
                    clarification_prompt = create_clarification_prompt(original_response, current_labels)
                    clarification_output = llm_client.generate_text(
                        clarification_prompt, max_new_tokens=30, temperature=0.1, use_chat_template=True
                    )
                    parsed, _ = parse_stance(clarification_output, current_labels)
                    tries += 1
                    if parsed != 'RERUN':
                        df.at[idx, 'predicted_stance'] = original_response
                        df.at[idx, 'parsed_output'] = parsed
                        df.at[idx, 'prompted_again'] = True
                        df.at[idx, 'number_tries'] = tries
                        print(f"✅ Clarification parsed: {parsed}")
                        continue

                # Reprompt with constraint
                prompt_with_constraint = inject_label_constraint(prompt, current_labels)
                
                for attempt in range(20):
                    print(f"Attempt {attempt + 1} for index {idx}")
                    output = llm_client.generate_text(
                        prompt_with_constraint, max_new_tokens=max_new_token, temperature=0.1, use_chat_template=False
                    )
                    
                    parsed, _ = parse_stance(output, current_labels)
                    tries += 1
                    if parsed != 'RERUN':
                        print(f"✅ Successfully parsed output: {parsed}")
                        df.at[idx, 'number_tries'] = tries
                        break

                # Fallback logic
                if parsed == 'RERUN':
                    fb = fallback_label(current_labels)
                    if fb:
                        parsed = fb
                        df.at[idx, 'fallback_used'] = True
                        print(f"🔁 Fallback to default label: {parsed}")

                df.at[idx, 'predicted_stance'] = output
                df.at[idx, 'parsed_output'] = parsed
                df.at[idx, 'prompted_again'] = True

        # Reorder columns to place 'parsed_output' after 'predicted_stance'
        try:
            cols = df.columns.tolist()
            pred_stance_idx = cols.index('predicted_stance')
            parsed_col = cols.pop(cols.index('parsed_output'))
            cols.insert(pred_stance_idx + 1, 'parsed_output')
            df = df[cols]
        except ValueError:
            print(f"Warning: 'predicted_stance' column not found. Appending new column to the end.")

        # Save
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path, index=False)
        print(f"✅ Processed and saved: {output_file_path}")

    except Exception as e:
        print(f"❌ Error processing file '{input_file_path}': {e}")


def process_all_files_with_lora(target_model_key=None, clarify: bool = False, lora_dir: str = LORA_DIR):
    """
    Process all CSV files in phase3_results with LoRA adapters.
    Groups files by target to minimize adapter loading.
    """
    print(f"🚀 Starting LoRA-enabled parsing process...")
    print(f"Input directory: '{INPUT_ROOT_DIR}'")
    print(f"Output directory: '{OUTPUT_ROOT_DIR}'")
    print(f"LoRA directory: '{lora_dir}'")
    if target_model_key:
        print(f"🎯 Target model: {target_model_key}")
    else:
        print(f"🌍 Processing all models")
    print(f"🧭 Clarifying prompt enabled: {clarify}")

    if not os.path.isdir(INPUT_ROOT_DIR):
        print(f"❌ Error: Input directory '{INPUT_ROOT_DIR}' not found. Exiting.")
        return

    for dirpath, dirnames, filenames in os.walk(INPUT_ROOT_DIR):
        dirnames[:] = [d for d in dirnames if d not in ['.ipynb_checkpoints']]
        model_key = os.path.basename(dirpath)

        if target_model_key and model_key != target_model_key:
            if model_key in MODEL_CONFIG:
                print(f"⏭️ Skipping model: {model_key} (target: {target_model_key})")
            continue

        if model_key not in MODEL_CONFIG:
            continue

        print(f"🔄 Processing model: {model_key}")
        max_new_token = max_new_token_config[model_key]

        # Group files by target to optimize LoRA loading
        files_by_target = {}
        for filename in filenames:
            if not filename.endswith('.csv'):
                continue
            
            try:
                target = extract_target_from_filename(filename, dirpath)
                if target not in files_by_target:
                    files_by_target[target] = []
                files_by_target[target].append(filename)
            except ValueError as e:
                print(f"❌❌❌❌❌❌❌ Error extracting target from filename '{filename}': {e} ❌❌❌❌❌❌❌")
                continue

        if not files_by_target:
            print(f"⚠️ No valid CSV files found in {dirpath}")
            continue

        # Process files grouped by target
        for target, target_files in files_by_target.items():
            # Clear CUDA cache before loading new adapter
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            adapter_path = os.path.join(lora_dir, model_key, target)
            if os.path.exists(adapter_path):
                print(f"🔄 Loading LoRA adapter for target '{target}' from {adapter_path}")
                llm_client = LLMClient(model_key, cache_dir=HF_CACHE_DIR, adapter_path=adapter_path)
            else:
                print(f"❌ ERROR: LoRA adapter not found at {adapter_path}")
                print(f"Available adapters in {os.path.join(lora_dir, model_key)}: {os.listdir(os.path.join(lora_dir, model_key)) if os.path.exists(os.path.join(lora_dir, model_key)) else 'Directory not found'}")
                continue  # Skip this target if adapter is missing
            
            print(f"📁 Processing {len(target_files)} files for target '{target}'")
            
            # Process all files for this target
            for filename in target_files:
                input_file_path = os.path.join(dirpath, filename)
                output_dir_path = dirpath.replace(INPUT_ROOT_DIR, OUTPUT_ROOT_DIR, 1)
                output_file_path = os.path.join(output_dir_path, filename)
                
                # Determine dataset name for labels
                dataset_name = None
                if 'in_target' in dirpath:
                    match = re.search(r'dataset-([a-zA-Z0-9]+)_', filename)
                    if match:
                        dataset_name = match.group(1)
                elif 'cross_target' in dirpath:
                    match = re.search(r'dst-([a-zA-Z0-9_]+)_', filename)
                    if match:
                        dst_target_abbr = match.group(1)
                        dataset_name = TARGET_DATASET_MAP.get(dst_target_abbr)
                        if not dataset_name:
                            print(f"❌ ERROR: Destination target '{dst_target_abbr}' not found in TARGET_DATASET_MAP for file '{filename}'")
                            continue
                
                if not dataset_name:
                    print(f"❌ ERROR: Could not determine dataset for '{filename}'. This should not happen!")
                    continue
                
                if dataset_name not in LABELS:
                    print(f"❌ ERROR: Dataset '{dataset_name}' not found in LABELS mapping for file '{filename}'")
                    continue
                
                current_labels = LABELS[dataset_name]
                process_single_file(input_file_path, output_file_path, llm_client, 
                                  max_new_token, clarify, current_labels)
            
            # Clean up after processing target
            del llm_client
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print(f"🧹 Cleared LoRA adapter for target '{target}' from memory.")

    print("\n✨ LoRA parsing complete.")


# --- Run the script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse stance labels from LoRA-generated LLM outputs.')
    parser.add_argument('--model', type=str, help='Specify a single model key to process (e.g., mistral_24b).')
    parser.add_argument('--clarify', action='store_true', default=False,
                        help='Use clarification prompt before rerun.')
    parser.add_argument("--lora-dir", type=str, default=LORA_DIR,
                        help='Directory containing LoRA adapters.')
    args = parser.parse_args()

    if args.model:
        process_all_files_with_lora(args.model, clarify=args.clarify, lora_dir=args.lora_dir)
    else:
        process_all_files_with_lora(clarify=args.clarify, lora_dir=args.lora_dir)
