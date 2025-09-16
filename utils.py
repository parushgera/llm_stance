
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Any
import pandas as pd
import random
import os
from mappings import TARGETS_MAP, KNOWLEDGE_BASE
SYSTEM_INSTRUCTION_GENERAL = "SYSTEM: You are an expert AI for stance detection. Your primary goal is to analyze text impartially and determine the author's precise stance according to the provided categories and instructions. You must strictly adhere to the defined output format, providing only the requested information. Focus on objective analysis of the text."


def count_prompt_tokens(prompt: str, tokenizer: AutoTokenizer) -> int:
    """
    Calculates the number of tokens in a given prompt using the provided tokenizer.

    Args:
        prompt (str): The text prompt.
        tokenizer (AutoTokenizer): The tokenizer instance loaded with the LLM.

    Returns:
        int: The number of tokens in the prompt.
    """
    # The tokenizer's encode method (or __call__) will convert text to token IDs.
    # The length of these IDs is the token count.
    # We don't need to add special tokens or truncate here for just counting,
    # but `return_tensors="pt"` is harmless and standard.
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs["input_ids"].shape[1]


def get_current_stance_labels(dataset_name: Optional[str]) -> List[str]:
    """Returns the appropriate list of stance labels based on the dataset name."""
    if dataset_name and 'wtwt' in dataset_name.lower():
        return ['COMMENT', 'SUPPORT', 'REFUTE', 'UNRELATED']
    elif dataset_name and 'pstance' in dataset_name.lower():
        return ['FAVOR', 'AGAINST']  # pstance only has FAVOR and AGAINST
    return ['AGAINST', 'FAVOR', 'NONE']  # Default for semeval and covid


def get_few_shot_examples(
    df: pd.DataFrame,
    n_examples_per_class: int,
    target_filter: Optional[str] = None,
    type_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None # This is the primary dataset name (e.g., 'semeval', 'wtwt')
) -> List[Dict[str, str]]:
    """
    Selects 'n' few-shot examples per stance class from the DataFrame, after applying specified filters.
    These examples will *only* include 'tweet' and 'stance'.
    """
    filtered_df = df.copy()

    # The dataset_filter here refers to the actual dataset name in your df['dataset'] column
    if dataset_filter:
        filtered_df = filtered_df[filtered_df['dataset'] == dataset_filter]
    
    if target_filter:
        # Assuming df['target'] contains the FULL NAMES (e.g., 'Donald Trump')
        global TARGETS_MAP
        target_full_name = TARGETS_MAP.get(target_filter, target_filter)
        filtered_df = filtered_df[filtered_df['target'] == target_full_name]
        
    if type_filter:
        filtered_df = filtered_df[filtered_df['type'] == type_filter]

    if filtered_df.empty:
        print(f"Warning: No data found for specified filters to select few-shot examples (Target={target_filter}, Dataset={dataset_filter}, Type={type_filter}).")
        return []

    # Get the appropriate stance labels for sampling based on the dataset_filter
    current_stance_labels = get_current_stance_labels(dataset_filter)

    all_few_shot_examples = []

    for stance_class in current_stance_labels: # Use dynamic labels for sampling
        class_data = filtered_df[filtered_df['stance'] == stance_class]
        
        num_available = len(class_data)
        if num_available == 0:
            print(f"Warning: No examples found for stance '{stance_class}' in dataset '{dataset_filter}' with current filters.")
            continue
        
        if num_available < n_examples_per_class:
            print(f"Warning: Only {num_available} examples available for stance '{stance_class}', requested {n_examples_per_class}. Using all available.")
            sampled_examples = class_data.to_dict(orient='records')
        else:
            sampled_examples = random.sample(class_data.to_dict(orient='records'), n_examples_per_class)

        for ex in sampled_examples:
            example_dict = {'tweet': ex['text'], 'stance': ex['stance']}
            all_few_shot_examples.append(example_dict)
            
    random.shuffle(all_few_shot_examples)

    return all_few_shot_examples
