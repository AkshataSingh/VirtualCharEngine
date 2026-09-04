import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()

messages = [{"role": "user", "content": "You are a friendly game character. Greet the player and ask how you can help them today."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
input_ids = inputs["input_ids"]

with torch.no_grad():
    gen_out = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

full_sequence = gen_out.sequences[0]
generated_part = full_sequence[input_ids.shape[1]:]
print("Generated tokens (via generate(), cached):", [tokenizer.decode([t]) for t in generated_part.tolist()])

step_index = 4  # the token right after "Hello! I'm" -> should be " here" in the true baseline
cached_logits = gen_out.scores[step_index][0]

prefix = full_sequence[: input_ids.shape[1] + step_index].unsqueeze(0)
with torch.no_grad():
    plain_out = model(input_ids=prefix)
plain_logits = plain_out.logits[0, -1, :]

def top5(logits):
    values, indices = torch.topk(logits, 5)
    return [(tokenizer.decode([idx.item()]), round(val.item(), 4)) for idx, val in zip(indices, values)]

print("\nTop-5 from generate() (cached):", top5(cached_logits))
print("Top-5 from plain forward pass (uncached):", top5(plain_logits))
print(f"\nArgmax match: {cached_logits.argmax().item() == plain_logits.argmax().item()}")
print(f"Max abs difference in logits: {(cached_logits - plain_logits).abs().max().item():.6f}")