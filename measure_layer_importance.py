import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

messages = [{"role": "user", "content": "You are a friendly game character. Greet the player and ask how you can help them today."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

def get_hidden_states():
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1]

reference = get_hidden_states()
num_layers = len(model.model.layers)
scores = []

for i in range(num_layers):
    original_forward = model.model.layers[i].forward
    model.model.layers[i].forward = lambda hidden_states, *args, **kwargs: hidden_states

    modified = get_hidden_states()
    model.model.layers[i].forward = original_forward

    sim = torch.nn.functional.cosine_similarity(reference.flatten(), modified.flatten(), dim=0).item()
    scores.append((i, sim))
    print(f"Layer {i}: similarity when skipped = {sim:.4f}")

scores.sort(key=lambda x: x[1], reverse=True)
print("\nSafest to remove first (highest similarity = least impact):")
for i, sim in scores:
    print(f"  Layer {i}: {sim:.4f}")