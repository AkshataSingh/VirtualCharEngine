import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SPARSITY = 0.35

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

parameters_to_prune = [
    (module, "weight")
    for module in model.modules()
    if isinstance(module, torch.nn.Linear)
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=SPARSITY,
)

for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)

model.save_pretrained("./qwen_unstructured_pruned")
tokenizer.save_pretrained("./qwen_unstructured_pruned")
print(f"Saved {SPARSITY*100:.0f}% unstructured-pruned model to ./qwen_unstructured_pruned")