import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor, DynamicCache

TARGET_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_PATH = "./qwen_pruned_distilled"

K = 4
MAX_NEW_TOKENS = 60

tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
target = AutoModelForCausalLM.from_pretrained(TARGET_PATH, dtype=torch.float32).eval()
draft = AutoModelForCausalLM.from_pretrained(DRAFT_PATH, dtype=torch.float32).eval()

rep_penalty = RepetitionPenaltyLogitsProcessor(penalty=1.1)

messages = [{"role": "user", "content": "You are a friendly game character. Greet the player and ask how you can help them today."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
input_ids = inputs["input_ids"]

generated = input_ids
num_generated = 0
total_accepted = 0
total_proposed = 0
target_forward_calls = 0
round_num = 0

draft_cache = DynamicCache()
target_cache = DynamicCache()

with torch.no_grad():
    draft_prime = draft(input_ids=generated, past_key_values=draft_cache, use_cache=True)
    draft_cache = draft_prime.past_key_values
    target_prime = target(input_ids=generated, past_key_values=target_cache, use_cache=True)
    target_cache = target_prime.past_key_values
    # "pending" logits = target's prediction for the token right after the last confirmed one
    pending_target_logits = target_prime.logits[:, -1, :]

start_time = time.perf_counter()

with torch.no_grad():
    while num_generated < MAX_NEW_TOKENS:
        round_num += 1

        # --- Draft proposes K tokens, incrementally, growing its own cache by 1 each step ---
        proposed_tokens = []
        last_token = generated[:, -1:]
        for _ in range(K):
            draft_out = draft(input_ids=last_token, past_key_values=draft_cache, use_cache=True)
            draft_cache = draft_out.past_key_values
            raw_logits = draft_out.logits[:, -1, :]
            processed_logits = rep_penalty(generated, raw_logits)
            next_token = processed_logits.argmax(dim=-1, keepdim=True)
            proposed_tokens.append(next_token)
            last_token = next_token

        proposed_tensor = torch.cat(proposed_tokens, dim=1)

        # --- Target verifies all K proposed tokens in ONE forward pass, reusing its existing cache ---
        target_out = target(input_ids=proposed_tensor, past_key_values=target_cache, use_cache=True)
        target_forward_calls += 1

        num_accepted = 0
        target_choices = []
        check_context = generated
        current_logits = pending_target_logits
        for i in range(K):
            processed_logits = rep_penalty(check_context, current_logits)
            target_next = processed_logits.argmax(dim=-1, keepdim=True)
            target_choices.append(target_next.item())
            if target_next.item() == proposed_tensor[0, i].item():
                num_accepted += 1
                check_context = torch.cat([check_context, proposed_tensor[:, i : i + 1]], dim=1)
                current_logits = target_out.logits[:, i, :]  # prediction for the NEXT position, to check proposed[i+1]
            else:
                break

        if round_num <= 3:
            draft_tok_strs = [tokenizer.decode([t]) for t in proposed_tensor[0].tolist()]
            target_tok_strs = [tokenizer.decode([t]) for t in target_choices]
            print(f"Round {round_num}: draft proposed {draft_tok_strs} | target would pick {target_tok_strs} | accepted={num_accepted}")

        total_proposed += K
        total_accepted += num_accepted

        # Trim both caches to only the confirmed length: len(generated) + num_accepted (+1 bonus token added below)
        confirmed_len = generated.shape[1] + num_accepted
        target_cache.crop(confirmed_len)
        draft_cache.crop(confirmed_len)

        if num_accepted > 0:
            generated = torch.cat([generated, proposed_tensor[:, :num_accepted]], dim=1)
            num_generated += num_accepted
            if (proposed_tensor[0, :num_accepted] == tokenizer.eos_token_id).any():
                break

        bonus_processed = rep_penalty(generated, current_logits)
        bonus_token = bonus_processed.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, bonus_token], dim=1)
        num_generated += 1
        if bonus_token.item() == tokenizer.eos_token_id:
            break

        # Only the target needs pre-priming (it processes a batch of K next round).
        # The draft naturally processes the bonus token as its own first step next round.
        target_step = target(input_ids=bonus_token, past_key_values=target_cache, use_cache=True)
        target_cache = target_step.past_key_values
        pending_target_logits = target_step.logits[:, -1, :]

end_time = time.perf_counter()

output_text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
total_time = end_time - start_time

print("\n--- Output ---")
print(output_text)
print(f"\nTotal time: {total_time:.2f} s")
print(f"Tokens generated: {num_generated}")
print(f"TPOT: {total_time / num_generated * 1000:.1f} ms/token")
print(f"Target model forward calls: {target_forward_calls}")
print(f"Acceptance rate: {total_accepted}/{total_proposed} ({total_accepted/total_proposed*100:.1f}%)")
print(f"Avg tokens per target forward call: {num_generated / target_forward_calls:.2f}")