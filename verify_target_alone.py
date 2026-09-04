import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()

messages = [{"role": "user", "content": "You are a friendly game character. Greet the player and ask how you can help them today."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)

output_text = tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("--- Target model alone (no speculative decoding) ---")
print(output_text)