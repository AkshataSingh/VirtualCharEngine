import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

img = Image.new("RGB", (300, 300), "white")
draw = ImageDraw.Draw(img)
draw.ellipse((75, 75, 225, 225), fill="red")
img.save("test_image.png")

print(f"Loading {MODEL_NAME} ... (larger model, expect a longer download/load than before)")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    device_map="cpu",
)
model.eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "test_image.png"},
            {"type": "text", "text": "What shape and color do you see in this image?"},
        ],
    }
]

text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[img], return_tensors="pt")

start_time = time.perf_counter()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50)
end_time = time.perf_counter()

new_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
output_text = processor.batch_decode(
    generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
)[0]

print("\n--- Output ---")
print(output_text)
print(f"\nTotal generation time: {end_time - start_time:.2f} s")
print(f"Tokens generated: {new_tokens}")
print(f"TPOT: {(end_time - start_time) / new_tokens * 1000:.1f} ms/token")