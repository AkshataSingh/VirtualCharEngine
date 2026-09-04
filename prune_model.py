import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS_TO_REMOVE = [18, 13, 12, 19,10]  # remove 4 of 24 layers (~17%)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Original layer count: {len(model.model.layers)}")

kept_layers = torch.nn.ModuleList(
    [layer for i, layer in enumerate(model.model.layers) if i not in LAYERS_TO_REMOVE]
)
model.model.layers = kept_layers
model.config.num_hidden_layers = len(kept_layers)

if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
    model.config.layer_types = [
        lt for i, lt in enumerate(model.config.layer_types) if i not in LAYERS_TO_REMOVE
    ]

print(f"Pruned layer count: {len(model.model.layers)}")

model.save_pretrained("./qwen_pruned")
tokenizer.save_pretrained("./qwen_pruned")
print("Saved pruned model to ./qwen_pruned")