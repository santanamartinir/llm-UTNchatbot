import os
import sys
import json
import re
import glob
import shutil
from openai import OpenAI

# Setup your DeepSeek API client
client = OpenAI(api_key="sk-638977128d82444d8bd5146fd9a90b02", base_url="https://api.deepseek.com")

def extract_useful_text(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text_sections = data['text_sections']
    relevant_text = [
        text_sections[sec].strip()
        for sec in text_sections
        if text_sections[sec].strip() and len(text_sections[sec].strip()) > 30
    ]
    return "\n\n".join(relevant_text)

def generate_qna_pairs(text):
    prompt = f"""
Your task is to generate realistic, high-quality question-answer pairs suitable for training a helpful chatbot that assists international students at the University of Technology Nuremberg (UTN).

Instructions:

- Generate a suitable number of diverse Q&A pairs (anything between 0 and 100, depending on the amount of information present) based on the provided text.
- The questions should reflect realistic queries students (especially international) might ask, using natural and varied phrasing.
- Include casual language or informal greetings occasionally to mimic real interactions.
- Answers should be concise (maximum 2–3 sentences), self-contained, and clearly understandable without additional context.
- Ensure answers directly utilize information provided in the text, without introducing details not mentioned.
- Cover multiple relevant topics present in the provided text.

Format:

Q1: [Realistic student question]
A1: [Concise, clear assistant response directly from the text]

Q2: [Follow-up or independent question]
A2: [Concise, clear assistant response directly from the text]

[Continue this format for all pairs]

Provided text:
---
{text}
---
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in generating clear and concise Q&A pairs."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=8192,
    )

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    print(f"API Call - Input tokens: {input_tokens}, Output tokens: {output_tokens}")

    return response.choices[0].message.content

def convert_to_jsonl(qna_text):
    qa_pairs = re.findall(r"Q\d+: (.+?)\nA\d+: (.+?)(?=\nQ|\Z)", qna_text, re.DOTALL)
    formatted_pairs = []
    for q, a in qa_pairs:
        formatted_pairs.append({
            "conversations": [
                {"role": "system", "content": "You are a knowledgeable assistant providing information about the University of Technology Nuremberg (UTN) (German: Technische Universität Nürnberg (UTN)), student life in Germany, and life in Nuremberg."},
                {"role": "user", "content": q.strip()},
                {"role": "assistant", "content": a.strip()}
            ]
        })
    return formatted_pairs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_json_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = os.path.join(input_folder, "Q_A")

    # Create or clear the Q_A folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Processing all JSON files
    for file_path in glob.glob(os.path.join(input_folder, "*.json")):
        print(f"Processing file: {os.path.basename(file_path)}")
        text = extract_useful_text(file_path)
        qna_text = generate_qna_pairs(text)
        jsonl_data = convert_to_jsonl(qna_text)

        output_file = os.path.basename(file_path).replace(".json", "_qa.jsonl")
        with open(os.path.join(output_folder, output_file), "w", encoding="utf-8") as f:
            for entry in jsonl_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

    print("✅ All files processed successfully!")