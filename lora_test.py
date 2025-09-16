import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import os

MODEL_CONFIG = {
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi": "microsoft/Phi-3-mini-128k-instruct",
    "mistral-24b": "mistralai/Mistral-Small-Instruct-2409"
}
TARGET_MODULES_MAP = {
    "llama3_8b": ["q_proj", "v_proj"],
    "mistral_7b": ["q_proj", "v_proj"],
    "mistral-24b": ["q_proj", "v_proj"],
    "phi": [ "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj"]  # For Phi-3
}
# --- User selection ---
selected_model_key = "phi"  # Change this to the model you want to fine-tune
model_name = MODEL_CONFIG[selected_model_key]
output_dir = f"/storage3-ciber/parush/lora/{selected_model_key}"
os.makedirs(output_dir, exist_ok=True)
# ----------------------

dataset_path = "dataset/all_combined.csv"
df_dataset = pd.read_csv(dataset_path)

target = "Atheism"           # e.g., "Atheism" or a key from TARGETS_MAP
dataset = "semeval"          # e.g., "semeval" or "wtwt"
type_split = "train"
filtered_df = df_dataset[
    (df_dataset["target"] == target) &
    (df_dataset["dataset"] == dataset) &
    (df_dataset["type"] == type_split)
]

stance_labels = ["FAVOR", "AGAINST", "NONE"]  # Adjust as needed
def make_prompt(row):
    return f"Tweet: {row['text']}\nWhat is the stance? Options: {', '.join(stance_labels)} Your output should only contain one of the options."

def create_vanilla_prompt(row: str, stance_labels) -> str:

    
        
    labels_str = ", ".join([f"**{label}**" for label in stance_labels])
    prompt = f"""
Analyze the following tweet and determine the author's stance.
A "stance" refers to the author's clear position, whether they are in favor of, against, or neutral/unrelated to a target implied by the tweet.
A Target can be an entity, organization, policy, person, etc.
The stance must be one of the following: {labels_str}.
Your output should be a single word, representing the determined stance.

Tweet: "{row['text']}"

Stance:
"""
    return prompt




finetune_df = pd.DataFrame({
    "prompt": filtered_df.apply(make_prompt, axis=1),
    "completion": filtered_df["stance"]
})

dataset = Dataset.from_pandas(finetune_df)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    cache_dir="/storage3-ciber/parush"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=TARGET_MODULES_MAP[selected_model_key],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

def preprocess(example):
    prompt = example["prompt"]
    completion = example["completion"]
    full_text = prompt + "\n" + completion
    return tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = dataset.map(preprocess, batched=False)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    output_dir=output_dir,
    save_strategy="no",  # Do not save intermediate checkpoints
    logging_steps=10,
    report_to="none"
)

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

trainer.train()

#Save only the final LoRA adapter
trainer.save_model(output_dir)
print(f"LoRA fine-tuned model saved to {output_dir}")

# --- Test the fine-tuned model ---
from peft import PeftModel, PeftConfig

print("Testing the fine-tuned model on a sample prompt:")
peft_config = PeftConfig.from_pretrained(output_dir)
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", cache_dir="/storage3-ciber/parush")
ft_model = PeftModel.from_pretrained(base_model, output_dir)
ft_model.eval()

sample_prompt = create_vanilla_prompt(filtered_df.iloc[0], stance_labels)
inputs = tokenizer(sample_prompt, return_tensors="pt").to(ft_model.device)
with torch.no_grad():
    outputs = ft_model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))