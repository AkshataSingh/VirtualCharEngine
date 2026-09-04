import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
STUDENT_PATH = "./qwen_pruned"
OUTPUT_PATH = "./qwen_pruned_distilled"

TEMPERATURE = 2.0
LEARNING_RATE = 1e-5
EPOCHS = 3

PROMPTS = [
    "You are a friendly game character. Greet the player and ask how you can help them today.",
    "You are a friendly game character. The player just found a hidden treasure chest. React with excitement.",
    "You are a friendly game character. The player asks for directions to the nearest village. Give simple directions.",
    "You are a friendly game character. The player says goodbye. Say farewell warmly.",
    "You are a friendly game character. Explain what quests are available today.",
]

print("Loading teacher and student...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_NAME, dtype=torch.float32)
student = AutoModelForCausalLM.from_pretrained(STUDENT_PATH, dtype=torch.float32)

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

student.train()
optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)

print("Generating teacher completions...")
training_examples = []
for prompt in PROMPTS:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        generated = teacher.generate(**inputs, max_new_tokens=60, do_sample=False)
    training_examples.append(generated)

print(f"Training on {len(training_examples)} examples for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    for full_ids in training_examples:
        with torch.no_grad():
            teacher_logits = teacher(input_ids=full_ids).logits

        student_logits = student(input_ids=full_ids).logits

        loss = F.kl_div(
            F.log_softmax(student_logits / TEMPERATURE, dim=-1),
            F.softmax(teacher_logits / TEMPERATURE, dim=-1),
            reduction="batchmean",
        ) * (TEMPERATURE ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} — avg loss: {total_loss/len(training_examples):.4f}")

student.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"Saved distilled student to {OUTPUT_PATH}")