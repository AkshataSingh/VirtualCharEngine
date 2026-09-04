import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model.save_pretrained("./qwen_vl_hf")
processor.save_pretrained("./qwen_vl_hf")
print("Saved to ./qwen_vl_hf")