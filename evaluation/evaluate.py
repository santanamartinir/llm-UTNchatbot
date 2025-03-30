import json
import math
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from prettytable import PrettyTable
import argparse
import random
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Load YAML config
# Argument parser to read config file from command line
parser = argparse.ArgumentParser(description="Evaluate a Qwen model using perplexity.")
parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
args = parser.parse_args()

# Load YAML config from argument
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Load model from path or hf
model_name = config["model_name"]
print(f'{config["pretrained"] = }')
if config["pretrained"]:
    model_name = config["model_name"]
    print(f"Loading pretrained model: {model_name} from Hugging Face...")
else:
    model_name = config["model_path"]
    print(f"Loading retrained model from: {model_name}...")

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset from path specified in config
dataset_path = config["dataset_file"]
print(f"Loading dataset from: {dataset_path}")

with open(dataset_path, "r") as f:
    dataset = json.load(f)

# Select random QA pairs based on `num_qa_pairs`
num_qa_pairs = config.get("num_qa_pairs", len(dataset))
random_qa_pairs = random.sample(dataset, min(num_qa_pairs, len(dataset)))

# Initialize ROUGE scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


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


# Initialize variables for BERTScore
references = []
candidates = []

# Table Setup
table = PrettyTable()
table.field_names = ["Question", "Generated Answer", "Perplexity", "BERTScore Precision", "BERTScore Recall", "BERTScore F1", "BLEU Score", "ROUGE-1 Precision", "ROUGE-1 Recall", "ROUGE-1 F1"]


# Run Evaluation
for item in random_qa_pairs:
    question = item["question"]
    reference_answer = item["answer"]
    generated_answer = generate_answer(question)
    perplexity = round(calculate_perplexity(generated_answer), 3)

    # Append to lists for BERTScore calculation
    references.append([reference_answer])
    candidates.append(generated_answer)

    # Compute BERTScore
    P, R, F1 = bert_score(candidates, references, lang='en', rescale_with_baseline=True, use_fast_tokenizer=True)

    # Compute BLEU Score
    reference_tokens = [ref.split() for ref in references[-1]]
    candidate_tokens = candidates[-1].split()
    bleu = sentence_bleu(reference_tokens, candidate_tokens)

    # Compute ROUGE Score
    rouge_scores = rouge_scorer.score(reference_answer, generated_answer)

    # Add row to table
    table.add_row([
        question,
        generated_answer,
        perplexity,
        round(P[0].item(), 3),
        round(R[0].item(), 3),
        round(F1[0].item(), 3),
        round(bleu, 3),
        round(rouge_scores['rouge1'].precision, 3),
        round(rouge_scores['rouge1'].recall, 3),
        round(rouge_scores['rouge1'].fmeasure, 3)
    ])



# Display Results
# print(table)

# Save Table to eval_table.txt
with open(config["result_file_path"], "w") as f:
    f.write(table.get_string())

