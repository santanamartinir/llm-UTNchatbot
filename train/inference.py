import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_models(base_model_name, lora_model_path, device):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)

    # Load fine-tuned model with LoRA weights
    lora_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    lora_model = PeftModel.from_pretrained(lora_model, lora_model_path)

    base_model.eval()
    lora_model.eval()

    return tokenizer, base_model, lora_model

def generate_response(model, tokenizer, user_input, device, max_length=512):
    system_prompt = ("You are a knowledgeable assistant providing information about the "
                     "University of Technology Nuremberg (UTN) (German: Technische Universität Nürnberg (UTN)), "
                     "student life in Germany, and life in Nuremberg.")

    formatted_input = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
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



def main():
    parser = argparse.ArgumentParser(description="Compare Base vs. LoRA-tuned Qwen responses")
    parser.add_argument('--base_model', type=str, required=True, help="Base model name, e.g., 'Qwen/Qwen2.5-0.5B-Instruct'")
    parser.add_argument('--lora_model', type=str, required=True, help="Path to LoRA fine-tuned model directory")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run models on ('cuda' or 'cpu')")

    args = parser.parse_args()

    tokenizer, base_model, lora_model = load_models(args.base_model, args.lora_model, args.device)

    print("Enter 'exit' to quit.")
    while True:
        user_input = input("\nYour prompt: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        formatted_input = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        base_response = generate_response(base_model, tokenizer, formatted_input, args.device)
        lora_response = generate_response(lora_model, tokenizer, formatted_input, args.device)

        print("\n--- Base Model Response ---")
        print(base_response)

        print("\n--- LoRA-tuned Model Response ---")
        print(lora_response)


if __name__ == "__main__":
    main()
