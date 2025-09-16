import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_config import MODEL_CONFIG
from in_target_prompts_v2 import chat_manager, SYSTEM_INSTRUCTION
from peft import PeftModel
from typing import Optional
import os


class LLMClient:
    def __init__(self, model_key: str, cache_dir: str = None, adapter_path: Optional[str] = None, **kwargs):
        """Enhanced LLM client with chat template support"""
        if model_key not in MODEL_CONFIG:
            raise ValueError(f"Model key '{model_key}' not found in MODEL_CONFIG.")

        self.model_key = model_key
        self.model_id = MODEL_CONFIG[model_key]
        print(f"Loading model: {self.model_id}...")
        self.cache_dir = cache_dir

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        
        # Enhanced memory optimization for large models
        is_large_model = "24b" in model_key.lower() or "27b" in model_key.lower()
        
        if is_large_model:
            print(f"Detected large model ({model_key}), applying multi-GPU optimizations...")
            # For 24B models, use more explicit memory management
            default_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            # Check if we have 2+ GPUs available
            if torch.cuda.device_count() >= 2:
                print(f"Using {torch.cuda.device_count()} GPUs for model distribution")
                # Let auto handle distribution but with explicit memory settings
                default_kwargs.update({
                    "max_memory": {i: "75GB" for i in range(torch.cuda.device_count())},
                })
        else:
            # Standard loading for smaller models
            default_kwargs = {
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto"
            }
        
        final_kwargs = {**default_kwargs, **kwargs}

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            **final_kwargs
        )
        if adapter_path and os.path.isdir(adapter_path):
            print(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        # Print device distribution for multi-GPU setups
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            print(f"Model distributed across devices: {self.model.hf_device_map}")
        else:
            print(f"Model '{self.model_id}' loaded successfully on device: {self.model.device}.")
        
        # Set max input length (rest remains the same)
        if hasattr(self.model.config, "max_position_embeddings") and self.model.config.max_position_embeddings > 0 and self.model.config.max_position_embeddings < 1e9:
            self.max_input_length = self.model.config.max_position_embeddings
            print(f"Using model's max_position_embeddings: {self.max_input_length}")
        elif self.tokenizer.model_max_length > 0 and self.tokenizer.model_max_length < 1e9:
            self.max_input_length = self.tokenizer.model_max_length
            print(f"Using tokenizer's model_max_length: {self.max_input_length}")
        else:
            self.max_input_length = 4096
            print(f"Warning: Could not reliably determine model_max_length. Defaulting to {self.max_input_length}.")

    def generate_text(self, user_content_or_full_prompt: str, max_new_tokens: int = 50, temperature: float = 0.1, top_p: float = 0.95, top_k: int = 50, use_chat_template: bool = True) -> str:
        """
        Generate text with optional chat template formatting
        
        Args:
            user_content_or_full_prompt: Either user content (if use_chat_template=True) or full prompt
            use_chat_template: If True, treat input as user content and apply chat template
        """
        if use_chat_template:
            # Format using chat template
            prompt = chat_manager.format_prompt(self.model_key, user_content_or_full_prompt, SYSTEM_INSTRUCTION)
        else:
            # Use as-is (backward compatibility)
            prompt = user_content_or_full_prompt
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_length).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        do_sample = temperature > 0.001
        temp_arg = temperature if do_sample else None

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temp_arg,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = out[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip() 