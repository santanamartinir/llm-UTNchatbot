#!/usr/bin/env python
import os
import argparse
import yaml
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

        all_input_ids = []
        all_labels = []

        for turn in conversation:
            role = turn["role"]
            content = f"<|im_start|>{role}\n{turn['content']}<|im_end|>\n"
            
            tokenized_turn = tokenizer(content, add_special_tokens=False).input_ids

            all_input_ids.extend(tokenized_turn)

            if role == "assistant":
                all_labels.extend(tokenized_turn)
            else:
                all_labels.extend([-100] * len(tokenized_turn))

        # Ensure consistent length
        max_len = config.get("max_length", 1024)
        all_input_ids = all_input_ids[:max_len]
        all_labels = all_labels[:max_len]

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        padding_length = max_len - len(all_input_ids)

        if padding_length > 0:
            all_input_ids += [pad_token_id] * padding_length
            all_labels += [-100] * padding_length

        return {
            "input_ids": torch.tensor(all_input_ids),
            "labels": torch.tensor(all_labels),
            "attention_mask": torch.tensor([1] * (max_len - padding_length) + [0] * padding_length)
        }

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=False)

    # --- Model Setup with LoRA ---
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    # Corrected model loading
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        attn_implementation="eager"  # Explicitly avoid SDPA conflicts
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        bias=config.get("lora_bias", "none"),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"])
    )

    # Wrap model with LoRA
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
        max_grad_norm=1.0,
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
