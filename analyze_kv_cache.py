import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()

messages = [{"role": "user", "content": "Tell me a short story about a brave knight and a dragon."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

past_kv = outputs.past_key_values
num_layers = len(past_kv)
sample_key = past_kv[0][0]

print(f"Number of layers with cache: {num_layers}")
print(f"Key tensor shape (one layer): {sample_key.shape}")
print(f"Key tensor dtype: {sample_key.dtype}")

total_bytes = 0
for layer_kv in past_kv:
    key, value = layer_kv
    total_bytes += key.numel() * key.element_size()
    total_bytes += value.numel() * value.element_size()

seq_len = sample_key.shape[2]
bytes_per_token = total_bytes / seq_len

print(f"\nPrompt length: {seq_len} tokens")
print(f"Total KV-cache size at this length: {total_bytes / 1024:.1f} KB")
print(f"KV-cache size per token: {bytes_per_token:.1f} bytes/token")

for target_len in [100, 1000, 10000]:
    projected_kb = bytes_per_token * target_len / 1024
    print(f"Projected KV-cache size at {target_len} tokens: {projected_kb:.1f} KB ({projected_kb/1024:.2f} MB)")

int8_bytes_per_token = bytes_per_token / 4
print(f"\nIf KV-cache were quantized to int8: {int8_bytes_per_token:.1f} bytes/token (4x reduction)")