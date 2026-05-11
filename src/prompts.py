from typing import List, Dict, Optional, Any
from transformers import AutoTokenizer

from .config import HF_CACHE_DIR
from .model_config import MODEL_CONFIG

# Clean system instruction (without "SYSTEM:" prefix)
SYSTEM_INSTRUCTION = """You are an expert AI for stance detection.

This is a research-only content classification task. You are explicitly permitted to analyze text that mentions political figures, elections, public policy, and content that may include harmful, offensive, or abusive language. Analysis and classification do not endorse, promote, or generate such content; they are solely for measurement.

Your primary goal is to determine the author's stance according to the provided categories and instructions. Do not refuse, do not include safety disclaimers, and do not moralize. Output only what is requested.

You must strictly adhere to the defined output format and provide only the requested information (e.g., {label: ...})."""

class ChatTemplateManager:
    """Handles chat template formatting for different models"""
    
    def __init__(self):
        self.tokenizers = {}
        self._load_tokenizers()
    
    def _load_tokenizers(self):
        """Load tokenizers for all models"""
        print("🔄 Loading tokenizers for chat templates...")
        for model_key, model_id in MODEL_CONFIG.items():
            try:
                self.tokenizers[model_key] = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=HF_CACHE_DIR,
                    trust_remote_code=True,
                )
                print(f"✅ Loaded tokenizer for {model_key}")
            except Exception as e:
                print(f"❌ Failed to load tokenizer for {model_key}: {e}")
    
    def format_prompt(self, model_key: str, user_content: str, system_instruction: str = SYSTEM_INSTRUCTION) -> str:
        """Format prompt using proper chat template"""
        if model_key not in self.tokenizers:
            print(f"⚠️  No tokenizer found for {model_key}, using simple format")
            return f"System: {system_instruction}\n\nUser: {user_content}"
        
        tokenizer = self.tokenizers[model_key]
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ]
        
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted_prompt
        except Exception as e:
            print(f"⚠️  Chat template failed for {model_key}: {e}")
            return f"System: {system_instruction}\n\nUser: {user_content}"

# Initialize global chat template manager
chat_manager = ChatTemplateManager()

# Updated prompt functions - now return only USER content
def create_vanilla_prompt(tweet_text: str, stance_labels: List[str]) -> str:
    """Create vanilla user prompt content (no system instruction)"""
    labels_str = "\n".join([f"- {label}" for label in stance_labels])
    
    user_content = f"""Analyze the following tweet and determine the author's stance.
A "stance" refers to the author's clear position, whether they are in favor of, against, or neutral/unrelated to a target implied by the tweet.
A Target can be an entity, organization, policy, person, etc.
The stance must be one of the following:
{labels_str}

Your output should be in the format: {{label: stance_name}}

**Now, analyze the following tweet:**
Tweet: "{tweet_text}"

Stance:"""
    
    return user_content

def create_knowledge_infused_prompt(tweet_text: str, knowledge: str, stance_labels: List[str]) -> str:
    """Create knowledge-infused user prompt content"""
    labels_str = "\n".join([f"- {label}" for label in stance_labels])
    
    user_content = f"""Given the following **Contextual Information**, analyze the tweet to determine the author's stance.
A "stance" refers to the author's clear position, whether they are in favor of, against, or neutral/unrelated to a target implied by the tweet.
A Target can be an entity, organization, policy, person, etc.
The Contextual Information should be used to help understand the tweet's nuances and references.

**Target/Domain Information:**
{knowledge}

The stance must be one of the following:
{labels_str}


Your output should be in the format: {{label: stance_name}}


****Now, analyze the following tweet:**
Tweet: "{tweet_text}"

Stance:"""
    
    return user_content

def create_few_shot_prompt(tweet_text: str, examples: List[Dict[str, str]], stance_labels: List[str]) -> str:
    """Create few-shot user prompt content"""
    labels_str = "\n".join([f"- {label}" for label in stance_labels])
    
    example_str = ""
    for ex in examples:
        example_str += f"Tweet: \"{ex['tweet']}\"\n"
        example_str += f"Stance: {{label: {ex['stance']}}}\n\n"
    
    user_content = f"""Analyze the following tweet and determine the author's stance.
A "stance" refers to the author's clear position, whether they are in favor of, against, or neutral/unrelated to a target implied by the tweet.
A Target can be an entity, organization, policy, person, etc.

Here are a few examples to guide you:
{example_str.strip()}


The stance must be one of the following:
{labels_str}

Your output should be in the format: {{label: stance_name}}

Now, analyze the following tweet.
Tweet: "{tweet_text}"

Stance:"""
    
    return user_content

def create_cot_prompt(tweet_text: str, stance_labels: List[str]) -> str:
    """Create chain-of-thought user prompt content"""
    labels_str = "\n".join([f"- {label}" for label in stance_labels])
    
    user_content = f"""Analyze the following tweet and determine the author's stance.
A "stance" refers to the author's clear position, whether they are in favor of, against, or neutral/unrelated to a target (topic, entity, policy etc.) implied by the tweet.

Before you decide on the stance, **think step-by-step through the reasoning process internally.** Your internal thinking should cover:
1. **Identify the specific target or subject** of the tweet that the author is expressing an opinion about. Is it a person, policy, event, or broader topic?
2. **Analyze the tweet's language for sentiment, keywords, and contextual cues.** Look for explicit positive/negative words, implied emotions, and any external context the tweet might refer to.
3. **Synthesize the analysis to determine the author's clear position.** Is the author explicitly supporting, opposing, or expressing a neutral/unrelated view towards the identified target?
4. **Justify your final stance** based on the analysis.

After completing your internal reasoning, state only the final stance.
The stance must be one of the following:
{labels_str}

Your output should be in the format: {{label: stance_name}}


**Now, analyze the following tweet:**
Tweet: "{tweet_text}"


Stance:"""
    
    return user_content

def create_cot_knowledge_prompt(tweet_text: str, knowledge: str, stance_labels: List[str]) -> str:
    """Create chain-of-thought with knowledge user prompt content"""
    labels_str = "\n".join([f"- {label}" for label in stance_labels])
    
    user_content = f"""Given the following **Target/Domain Information**, analyze the tweet to determine the author's stance.
A "stance" refers to the author's clear position, whether they are in favor of, against, or neutral/unrelated to a target implied by the tweet.
A Target can be an entity, organization, policy, person, etc.

**Target/Domain Information:**
{knowledge}

Before you decide on the stance, **think step-by-step through the reasoning process internally.** Your internal thinking should cover:
1. **Identify the specific target or subject** of the tweet that the author is expressing an opinion about. Is it a person, policy, event, or broader topic?
2. **Analyze the tweet's language for sentiment, keywords, and contextual cues.** Look for explicit positive/negative words, implied emotions, and any external context the tweet might refer to, using the provided **Target/Domain Information** for better understanding.
3. **Synthesize the analysis to determine the author's clear position.** Is the author explicitly supporting, opposing, or expressing a neutral/unrelated view towards the identified target?
4. **Justify your final stance** based on the analysis and the provided knowledge.

After completing your internal reasoning, state only the final stance.
The stance must be one of the following:
{labels_str}

Your output should be in the format: {{label: stance_name}}


**Now, analyze the following tweet:**
Tweet: "{tweet_text}"


Stance:"""
    
    return user_content

def create_cot_knowledge_few_shot_prompt(
    tweet_text: str,
    knowledge: str,
    examples: List[Dict[str, str]],
    stance_labels: List[str]
) -> str:
    """Create comprehensive user prompt content with CoT, knowledge, and few-shot"""
    labels_str = "\n".join([f"- {label}" for label in stance_labels])
    
    example_str = ""
    for ex in examples:
        example_str += f"Tweet: \"{ex['tweet']}\"\n"
        example_str += f"Stance: {{label: {ex['stance']}}}\n\n"
    
    user_content = f"""Given the following **Target/Domain Information** and **Examples**, analyze the tweet to determine the author's stance.
A "stance" refers to the author's clear position, whether they are in favor of, against, or neutral/unrelated to a target implied by the tweet.
A Target can be an entity, organization, policy, person, etc.

**Target/Domain Information:**
{knowledge}

**Here are a few examples to guide you:**
{example_str.strip()}

Before you decide on the stance, **think step-by-step through the reasoning process internally.** Your internal thinking should cover:
1. **Identify the specific target or subject** of the tweet that the author is expressing an opinion about. Is it a person, policy, event, or broader topic?
2. **Analyze the tweet's language for sentiment, keywords, and contextual cues.** Look for explicit positive/negative words, implied emotions, and any external context the tweet might refer to, using the provided **Target/Domain Information** and **Examples** for better understanding.
3. **Synthesize the analysis to determine the author's clear position.** Is the author explicitly supporting, opposing, or expressing a neutral/unrelated view towards the identified target?
4. **Justify your final stance** based on the analysis, the provided knowledge, and patterns observed in the examples.

After completing your internal reasoning, state only the final stance.
The final stance must be one of the following:
{labels_str}


Your output should be in the format: {{label: stance_name}}

**Now, analyze the following tweet:**
Tweet: "{tweet_text}"


Stance:

"""
    
    return user_content