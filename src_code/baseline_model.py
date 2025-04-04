from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Load dataset as prompt prefix
with open('datasets/q_a.json', 'r') as file:
    qa_data = json.load(file)

# with open('datasets/q_a_2.json', 'r') as file:
#     qa_data_2 = json.load(file)

prompt_prefix = "The following are examples of questions and answers regarding university and student life:\n"
prompt_prefix += "\n".join(
    [f"Q: {item['question']}\nA: {item['answer']}" for item in qa_data]  # + qa_data_2
)
prompt_prefix += "\nNow please answer the user's question *shortly, once* in about 200 words max. Respond in JSON format like: {\"answer\": \"<your short answer>\"}.\n"


def generate_response(prompt):
    # inputs = tokenizer(prompt, return_tensors="pt")
    # output = model.generate(**inputs, max_length=200)
    # return tokenizer.decode(output[0], skip_special_tokens=True)
    full_prompt = prompt_prefix + f"Q: {prompt}\nA:"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=inputs.input_ids.shape[-1] + 200, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Q: " in decoded_output and "\nA:" in decoded_output:
        decoded_output = decoded_output.split("\nA:")[-1].strip()

    try:
        json_start = decoded_output.index("{")
        json_str = decoded_output[json_start:]
        return json.loads(json_str)["answer"]
    except Exception as e:
        print(f"Failed to extract JSON answer: {e}")
        return decoded_output.strip()
    # Extract only generated answer
    # generated_answer = decoded_output.split(f"Q: {prompt}\nA:")[-1].strip()
    # print(generated_answer)
    # return generated_answer


# Example Question
# question = "What is the UTN University of Technology Nuremberg?"
# print(generate_response(question))
