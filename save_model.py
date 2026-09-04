from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.save_pretrained("./qwen_hf")
tokenizer.save_pretrained("./qwen_hf")
print("Saved to ./qwen_hf")