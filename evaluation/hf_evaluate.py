import json
import math
import torch
import yaml
import argparse
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prettytable import PrettyTable
# import evaluate
from evaluate import load

# Argument parser to read config file from command line
parser = argparse.ArgumentParser(description="Evaluate a Qwen model using perplexity, BERTScore, BLEU, and ROUGE.")
parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
parser.add_argument('--device', type=str, default='cuda', help="Device to run models on ('cuda' or 'cpu')")

args = parser.parse_args()
device = args.device

# Load YAML config from argument
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Load model from path or hf
model_name = config["model_name"]
print(f'{config["pretrained"] = }')
if config["pretrained"]:
    model_name = config["model_name"]
    print(f"Loading pretrained model: {model_name} from Hugging Face...")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
else:
    base_model_name = config['model_name']
    lora_model_path = config['model_path']
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load fine-tuned model with LoRA weights
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, lora_model_path) 
    model.eval()

# Load model & tokenizer


# Load dataset from path specified in config
dataset_path = config["dataset_file"]
print(f"Loading dataset from: {dataset_path}")

with open(dataset_path, "r") as f:
    dataset = json.load(f)

# Select random QA pairs based on `num_qa_pairs`
num_qa_pairs = config.get("num_qa_pairs", len(dataset))
# random_qa_pairs = random.sample(dataset, min(num_qa_pairs, len(dataset))) #random
random_qa_pairs = dataset[:min(num_qa_pairs, len(dataset))] # first 10

# Initialize Hugging Face evaluation metrics
rouge = load("rouge")
bleu = load("bleu")
bertscore = load("bertscore")

# Function to generate answer from Qwen model
def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Function to compute Perplexity
def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return math.exp(loss)

# Initialize variables for BLEU, ROUGE, and BERTScore
references = []
candidates = []

# Table Setup with updated columns for all ROUGE metrics
table = PrettyTable()
table.field_names = [
    "Question",
    "Generated Answer",
    # "Perplexity",
    "BERTScore Precision",
    "BERTScore Recall",
    "BERTScore F1",
    "BLEU Score",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "ROUGE-Lsum"
]

# Evaluate each QA pair individually
for item in random_qa_pairs:
    question = item["question"]
    reference_answer = item["answer"]
    generated_answer = generate_answer(question)
    perplexity = round(calculate_perplexity(generated_answer), 3)
    
    # Compute BERTScore for this sample
    bert_results = bertscore.compute(
        predictions=[generated_answer],
        references=[[reference_answer]],
        model_type="distilbert-base-uncased"
    )
    B_P = round(bert_results["precision"][0], 3)
    B_R = round(bert_results["recall"][0], 3)
    B_F1 = round(bert_results["f1"][0], 3)
    
    # Compute BLEU Score for this sample
    bleu_result = bleu.compute(
        predictions=[generated_answer],
        references=[[reference_answer]]
    )
    bleu_score = round(bleu_result["bleu"], 3)
    
    # Compute ROUGE Score for this sample
    # Note: evaluate's "rouge" returns a dict with keys "rouge1", "rouge2", "rougeL", "rougeLsum"
    rouge_result = rouge.compute(
        predictions=[generated_answer],
        references=[[reference_answer]]
    )
    # Convert each numpy.float64 to a float
    rouge1 = round(float(rouge_result["rouge1"]), 3)
    rouge2 = round(float(rouge_result["rouge2"]), 3)
    rougeL = round(float(rouge_result["rougeL"]), 3)
    rougeLsum = round(float(rouge_result["rougeLsum"]), 3)
    
    # Add row with all evaluation metrics to table
    table.add_row([
        question,
        generated_answer,
        # perplexity,
        B_P,
        B_R,
        B_F1,
        bleu_score,
        rouge1,
        rouge2,
        rougeL,
        rougeLsum
    ])

# Save Table to eval_table.txt
with open(config["result_file_path"], "w") as f:
    f.write(table.get_string())
