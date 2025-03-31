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
# def generate_answer(question):
#     inputs = tokenizer(question, return_tensors="pt")
#     with torch.no_grad():
#         output = model.generate(**inputs, max_new_tokens=50)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_response(model, tokenizer, user_input, device, max_length=512):
    system_prompt = (
                        "You are a helpful and knowledgeable assistant from the University of Technology Nuremberg (UTN) (German: Technische Universität Nürnberg (UTN)). "
                        "You provide information about the university, student life in Germany, and life in Nuremberg."
                        "Answer the student's question briefly (max 50 words), clearly, and professionally. "
                        "Provide direct answers only, no additional questions or examples.\n\n"
                    )

    formatted_input = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(formatted_input, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=False,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant response
    assistant_response = response.split("<|im_start|>assistant")[-1].strip()
    return assistant_response

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

# Initialize variables for BLEU, ROUGE, and BERTScore
total_perplexity = 0
total_bert_p = 0
total_bert_r = 0
total_bert_f1 = 0
total_bleu = 0
total_rouge1 = 0
total_rouge2 = 0
total_rougel = 0
total_rougel_sum = 0

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
    generated_answer = generate_response(model, tokenizer, question, device)  # Use generate_response here
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

    total_perplexity += perplexity
    total_bert_p += B_P
    total_bert_r += B_R
    total_bert_f1 += B_F1
    total_bleu += bleu_score
    total_rouge1 += rouge1
    total_rouge2 += rouge2
    total_rougel += rougeL
    total_rougel_sum += rougeLsum

# Calculate averages for each metric
num_samples = len(random_qa_pairs)
avg_perplexity = round(total_perplexity / num_samples, 3)
avg_bert_p = round(total_bert_p / num_samples, 3)
avg_bert_r = round(total_bert_r / num_samples, 3)
avg_bert_f1 = round(total_bert_f1 / num_samples, 3)
avg_bleu = round(total_bleu / num_samples, 3)
avg_rouge1 = round(total_rouge1 / num_samples, 3)
avg_rouge2 = round(total_rouge2 / num_samples, 3)
avg_rougel = round(total_rougel / num_samples, 3)
avg_rougel_sum = round(total_rougel_sum / num_samples, 3)

# Add a row for average scores
table.add_row([
    "Average Scores", 
    "-", 
    # avg_perplexity, 
    avg_bert_p, 
    avg_bert_r, 
    avg_bert_f1, 
    avg_bleu, 
    avg_rouge1, 
    avg_rouge2, 
    avg_rougel, 
    avg_rougel_sum
])


# Save Table to eval_table.txt
with open(config["result_file_path"], "w") as f:
    f.write(table.get_string())
