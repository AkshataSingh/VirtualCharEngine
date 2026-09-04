import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    device_map="cpu",
)
model.eval()

messages = [
    {"role": "user", "content": "You are a friendly game character. Greet the player and ask how you can help them today."}
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)

streamer = TextIteratorStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(**inputs, streamer=streamer,
                         max_new_tokens=100, do_sample=False)

start_time = time.perf_counter()
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

first_token_time = None
token_count = 0
output_text = ""

for token in streamer:
    now = time.perf_counter()
    if first_token_time is None:
        first_token_time = now
    token_count += 1
    output_text += token

thread.join()
end_time = time.perf_counter()

ttft = first_token_time - start_time
total_time = end_time - start_time
tpot = (total_time - ttft) / max(token_count - 1, 1)

print("\n--- Output ---")
print(output_text)
print("\n--- Baseline Latency ---")
print(f"Time-To-First-Token (TTFT): {ttft*1000:.1f} ms")
print(f"Time-Per-Output-Token (TPOT): {tpot*1000:.1f} ms/token")
print(f"Total tokens generated: {token_count}")
print(f"Total generation time: {total_time:.2f} s")
