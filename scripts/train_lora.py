"""Train one LoRA adapter per target/domain for each base model."""

import gc
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DATASET_PATH, HF_CACHE_DIR, LORA_DIR
from src.mappings import TARGET_DATASET_MAP, TARGET_MODULES_MAP, TARGETS_MAP
from src.model_config import MODEL_CONFIG
from src.prompts import SYSTEM_INSTRUCTION, chat_manager

# All 13 targets + 2 cross-domain WT-WT splits, in the order used by the paper.
ALL_TARGET_KEYS = [
    "ent", "hlt",                              # cross-domain (WT-WT)
    "bernie", "joe", "dtp",                    # P-Stance
    "dt", "la", "cc", "fm", "hc", "at",        # SemEval
    "face", "fauci", "school", "stay",         # COVID-19
]
warnings.filterwarnings("ignore")

log_dir = os.path.join(LORA_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

# Create a timestamp for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will also print to console
    ]
)

logger = logging.getLogger(__name__)

# Set environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logger.info(f"Loading dataset from: {DATASET_PATH}")
df_dataset = pd.read_csv(DATASET_PATH)
logger.info(f"Dataset loaded successfully. Total samples: {len(df_dataset)}")

total_combinations = len(MODEL_CONFIG) * len(ALL_TARGET_KEYS)
current_combination = 0

def cleanup_memory():
    """Aggressive memory cleanup function"""
    logger.info("Performing aggressive memory cleanup")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")

def get_dynamic_config(target_key, model_key):

    """Get dynamic LoRA and training configuration based on target dataset
    Used paper: https://arxiv.org/pdf/2402.17193
    Lora paper: https://arxiv.org/pdf/2106.09685v1/1000
    """
    dataset = TARGET_DATASET_MAP.get(target_key, 'unknown')
    
    # Reduce batch sizes for large models
    is_large_model = "24b" in model_key.lower() or "27b" in model_key.lower()
    # base_batch_size = 4 if is_large_model else 8  # Smaller for large models
    # pstance_batch_size = 8 if is_large_model else 16
    # wtwt_batch_size = 8 if is_large_model else 16

    base_batch_size = 8  # Smaller for large models
    pstance_batch_size = 16
    wtwt_batch_size = 16
    
    if dataset in ['semeval', 'covid']:
        # Configuration for Semeval and Covid targets
        lora_config = {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,  # Higher dropout for small datasets
            'bias': "none",
            'task_type': TaskType.CAUSAL_LM
        }
        training_config = {
            'per_device_train_batch_size': base_batch_size,
            'gradient_accumulation_steps': 4 if is_large_model else 2,  # Effective batch = 16
            'num_train_epochs': 5,  # More epochs for small dataset
            'learning_rate': 5e-4,  # Intermediate learning rate
            'fp16': True,
            'save_strategy': "no",
            'logging_steps': 10,
            'report_to': "none",
            'warmup_ratio': 0.1
        }
        logger.info(f"Using Semeval/Covid configuration for {target_key} (dataset: {dataset})")
        
    elif dataset == 'pstance':
        # Configuration for Pstance targets
        lora_config = {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': TaskType.CAUSAL_LM
        }
        training_config = {
            'per_device_train_batch_size': pstance_batch_size,
            'gradient_accumulation_steps': 4 if is_large_model else 2,  # Effective batch = 32
            'num_train_epochs': 3,
            'learning_rate': 3e-4,  # Original setting, solid default
            'fp16': True,
            'save_strategy': "no",
            'logging_steps': 10,
            'report_to': "none",
            'warmup_ratio': 0.1
        }
        logger.info(f"Using Pstance configuration for {target_key} (dataset: {dataset})")
        
    elif dataset == 'wtwt':
        # Configuration for WTWT targets
        lora_config = {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': TaskType.CAUSAL_LM
        }
        training_config = {
            'per_device_train_batch_size': wtwt_batch_size,
            'gradient_accumulation_steps': 4 if is_large_model else 2,  # Effective batch = 32
            'num_train_epochs': 2,  # Fewer epochs needed with larger data
            'learning_rate': 3e-4,
            'fp16': True,
            'save_strategy': "no",
            'logging_steps': 10,
            'report_to': "none",
            'warmup_ratio': 0.1
        }
        logger.info(f"Using WTWT configuration for {target_key} (dataset: {dataset})")
        
    else:
        # Default configuration for unknown datasets
        logger.warning(f"Unknown dataset '{dataset}' for target '{target_key}', using default configuration")
        lora_config = {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'bias': "none",
            'task_type': TaskType.CAUSAL_LM
        }
        training_config = {
            'per_device_train_batch_size': 16,
            'gradient_accumulation_steps': 2,
            'num_train_epochs': 3,
            'learning_rate': 3e-4,
            'fp16': True,
            'save_strategy': "no",
            'logging_steps': 10,
            'report_to': "none",
            'warmup_ratio': 0.1
        }
    
    return lora_config, training_config

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train LoRA adapters for stance detection')
    parser.add_argument('--models', type=str, nargs='+', choices=list(MODEL_CONFIG.keys()), 
                       help='Models to train (e.g., mistral_24b llama3_8b). Can specify multiple models separated by spaces.')
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# Determine which models to train
if args.models:
    models_to_train = {key: MODEL_CONFIG[key] for key in args.models}
    logger.info(f"Training models: {', '.join(args.models)}")
else:
    models_to_train = MODEL_CONFIG
    logger.info("Training all models")




for model_key, model_name in models_to_train.items():
    logger.info(f"Starting training for model: {model_key} ({model_name})")
    
    for target_key in ALL_TARGET_KEYS:
        current_combination += 1
        logger.info(f"Progress: {current_combination}/{total_combinations} - Training {model_key} for target {target_key}")
        
        output_dir = os.path.join(LORA_DIR, model_key, target_key)
        
        # Check if model already exists and has been trained
        if os.path.exists(output_dir):
            logger.info(f"Skipping {model_key}/{target_key}: Model already trained (found adapter_model.safetensors)")
            continue
        
        # Clean up memory before starting new training
        cleanup_memory()
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created: {output_dir}")
        
        filtered_df = df_dataset[df_dataset["target"] == TARGETS_MAP[target_key]]
        if filtered_df.empty:
            logger.warning(f"Skipping {model_key} {target_key}: no data found for target {TARGETS_MAP[target_key]}")
            continue

        logger.info(f"Found {len(filtered_df)} samples for target {target_key}")

        # Fix stance labels based on dataset and target
        dataset = TARGET_DATASET_MAP.get(target_key, 'unknown')
        if dataset == 'wtwt':
            stance_labels = ["COMMENT", "REFUTE", "SUPPORT", "UNRELATED"]
        elif dataset == 'pstance':
            stance_labels = ["FAVOR", "AGAINST"]  # P-stance only has FAVOR and AGAINST
        else:
            # For semeval and covid datasets
            stance_labels = ["FAVOR", "AGAINST", "NONE"]

        def make_training_example(row, model_key):
            # Create user content similar to your vanilla prompt style
            labels_str = ", ".join(stance_labels)
            user_content = f"""Analyze the following tweet and determine the author's stance.
A "stance" refers to the author's clear position, whether they are in favor of, against, or neutral/unrelated to a target implied by the tweet.

The stance must be one of the following: {labels_str}

Your output should be in the format: {{label: stance_name}}

Tweet: "{row['text']}"

Stance:"""
            
            # Create the full conversation
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"{{label: {row['stance']}}}"}
            ]
            
            # Format the entire conversation
            if model_key not in chat_manager.tokenizers:
                logger.warning(f"No tokenizer found for {model_key}, using simple format")
                return f"System: {SYSTEM_INSTRUCTION}\n\nUser: {user_content}\n\nAssistant: {{label: {row['stance']}}}"
            
            tokenizer = chat_manager.tokenizers[model_key]
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as e:
                logger.warning(f"Chat template failed for {model_key}: {e}, using simple format")
                return f"System: {SYSTEM_INSTRUCTION}\n\nUser: {user_content}\n\nAssistant: {{label: {row['stance']}}}"

        # Create dataset with full conversation format
        finetune_df = pd.DataFrame({
            "text": filtered_df.apply(lambda row: make_training_example(row, model_key), axis=1)
        })
        dataset = Dataset.from_pandas(finetune_df)
        logger.info(f"Dataset prepared with {len(dataset)} samples")

        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        def preprocess(example):
            # Now we just tokenize the full conversation text
            return tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=512  # Increased from 256 to accommodate full conversation
            )

        logger.info(f"Loading base model: {model_name}")

        # For 24B models, use more of the available H100 memory
        if "24b" in model_key.lower() or "27b" in model_key.lower():
            logger.info("Loading large model with multi-GPU optimization")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=HF_CACHE_DIR,
                low_cpu_mem_usage=True,
                max_memory={0: "75GB", 1: "75GB"},  # Use most of the 80GB per H100
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=HF_CACHE_DIR,
            )
        # --- START OF FIX ---
# Enable gradient checkpointing and prepare inputs BEFORE applying PEFT
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
# --- END OF FIX ---
        # Get dynamic configuration based on target
        lora_config_params, training_config_params = get_dynamic_config(target_key, model_key)
        
        logger.info("Configuring LoRA with dynamic parameters")
        lora_config = LoraConfig(
            target_modules=TARGET_MODULES_MAP[model_key],
            **lora_config_params
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA configured with target modules: {TARGET_MODULES_MAP[model_key]}")
        # ADD THIS DEBUGGING SECTION:
        
        
        
        logger.info(f"LoRA parameters: r={lora_config_params['r']}, alpha={lora_config_params['lora_alpha']}, dropout={lora_config_params['lora_dropout']}")
        
        logger.info("Tokenizing dataset")
        tokenized_dataset = dataset.map(preprocess, batched=False)
        logger.info("Dataset tokenization completed")
        
        logger.info("Setting up training arguments with dynamic parameters")
        training_args = TrainingArguments(
            output_dir=output_dir,
            **training_config_params
        )
        logger.info(f"Training arguments configured: batch_size={training_config_params['per_device_train_batch_size']}, "
                   f"epochs={training_config_params['num_train_epochs']}, lr={training_config_params['learning_rate']}")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        # Enable gradient checkpointing for large models
        if "24b" in model_key.lower() or "27b" in model_key.lower():
            trainer.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for large model")

        # Record training start time
        start_time = time.time()
        start_datetime = datetime.now()
        logger.info(f"Starting training for {model_key}/{target_key} at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Train and get history
            train_result = trainer.train()
            logger.info("Training completed successfully")
            
            # Save training history to CSV
            if hasattr(train_result, 'history') and train_result.history:
                history_df = pd.DataFrame(train_result.history)
                history_file = os.path.join(output_dir, "training_history.csv")
                history_df.to_csv(history_file, index=False)
                logger.info(f"Training history saved to: {history_file}")
            
        except Exception as e:
            logger.error(f"Training failed for {model_key}/{target_key}: {str(e)}")
            # Clean up memory after training failure
            del model, trainer, tokenized_dataset, dataset
            cleanup_memory()
            continue
        
        # Calculate training time
        end_time = time.time()
        end_datetime = datetime.now()
        training_time_seconds = end_time - start_time
        training_time_minutes = training_time_seconds / 60
        
        logger.info(f"Training completed in {training_time_minutes:.2f} minutes")
        
        # Save training time information
        training_time_file = os.path.join(output_dir, "training_time.txt")
        with open(training_time_file, "w") as f:
            f.write(f"Model: {model_key}\n")
            f.write(f"Target: {target_key}\n")
            f.write(f"Dataset: {TARGET_DATASET_MAP.get(target_key, 'unknown')}\n")
            f.write(f"Training started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training ended: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {training_time_seconds:.2f} seconds ({training_time_minutes:.2f} minutes)\n")
            f.write(f"Dataset size: {len(filtered_df)} samples\n")
            f.write(f"Model name: {model_name}\n")
            f.write(f"LoRA config: r={lora_config_params['r']}, alpha={lora_config_params['lora_alpha']}, dropout={lora_config_params['lora_dropout']}\n")
            f.write(f"Training config: batch_size={training_config_params['per_device_train_batch_size']}, epochs={training_config_params['num_train_epochs']}, lr={training_config_params['learning_rate']}\n")
        
        logger.info(f"Training time saved to: {training_time_file}")
        
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        logger.info(f"LoRA fine-tuned model saved to {output_dir}")

        # Aggressive memory cleanup
        logger.info("Performing aggressive memory cleanup")
        del model, trainer, tokenized_dataset, dataset, finetune_df
        cleanup_memory()
        logger.info(f"Completed training for {model_key}/{target_key}")

logger.info("All training completed!")
logger.info(f"Log file location: {log_file}")