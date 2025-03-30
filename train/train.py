#!/usr/bin/env python
import os
import argparse
import yaml
import torch

# Make sure to set this at the very beginning
os.environ["PYTORCH_ENABLE_SDPA"] = "0"

# ----- Monkey-Patch to handle extra dimension in SDPA -----
import transformers.integrations.sdpa_attention as sdpa_attention

def patched_repeat_kv(hidden_states, num_key_value_groups):
    # If hidden_states is 5-dimensional, assume shape: (batch, extra, num_key_value_heads, slen, head_dim)
    if hidden_states.dim() == 5:
        extra = hidden_states.size(1)
        if extra == 1:
            hidden_states = hidden_states.squeeze(1)
        else:
            batch, extra, n_heads, slen, head_dim = hidden_states.shape
            hidden_states = hidden_states.view(batch * extra, n_heads, slen, head_dim)
    # Now assume hidden_states is 4D
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    return hidden_states

sdpa_attention.repeat_kv = patched_repeat_kv
# -----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen with LoRA on UTN data")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML configuration file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse command-line arguments and load config
    args = parse_args()
    config = load_config(args.config)

    # --- Dataset and Tokenizer ---
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load dataset from the JSON file (config["dataset_file"] should point to your dataset file)
    dataset = load_dataset('json', data_files=config["dataset_file"])
    

    print("A")
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    print("B")

    def tokenize_function(examples):
        conversation = examples["conversations"]
        text = ""
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=config.get("max_length", 1024),
            return_tensors="pt"
        )

        # Mask labels for everything except assistant response
        labels = tokenized["input_ids"].clone()
        current_idx = 0
        labels_masked = [-100] * labels.size(1)  # Mask all tokens initially

        for turn in conversation:
            role = turn["role"]
            content = f"<|im_start|>{role}\n{turn['content']}<|im_end|>\n"
            tokenized_turn = tokenizer(content, return_tensors="pt").input_ids.size(1)
            
            if role == "assistant":
                labels_masked[current_idx:current_idx+tokenized_turn] = labels[0, current_idx:current_idx+tokenized_turn]
            
            current_idx += tokenized_turn

        tokenized["labels"] = torch.tensor([labels_masked])
        return tokenized


    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=False)

    # --- Model Setup with LoRA ---
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    print("C")

    # Disable SDPA / sliding window attention via config flags if present
    if hasattr(model.config, "use_sdpa"):
        model.config.use_sdpa = False
    if hasattr(model.config, "use_sliding_window_attention"):
        model.config.use_sliding_window_attention = False
    if hasattr(model.config, "sliding_window_attention"):
        model.config.sliding_window_attention = False

    # Create LoRA configuration from config parameters
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        bias=config.get("lora_bias", "none"),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"])
    )

    # Wrap the model with LoRA
    model = get_peft_model(model, lora_config)

    # --- Training Setup ---
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        evaluation_strategy=config.get("evaluation_strategy", "no"),
        learning_rate=config.get("learning_rate", 2e-4),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        num_train_epochs=config.get("num_train_epochs", 1),
        weight_decay=config.get("weight_decay", 0.01),
        save_total_limit=config.get("save_total_limit", 1),
        logging_dir=config.get("logging_dir", "./logs"),
        logging_steps=config.get("logging_steps", 1),
        do_eval=config.get("do_eval", False),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # --- Training ---
    trainer.train()

    # --- Save the Fine-Tuned Model and Tokenizer ---
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()
