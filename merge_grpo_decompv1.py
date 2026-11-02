"""
merge_grpo_decompv1.py - Merge the decomp_v1-based GRPO LoRA adapter into
decomp_v1's merged model (same pattern as merge_grpo.py, which does this for
the separate exp1-based GRPO run — kept as a distinct script/output dir so
neither run's artifacts overwrite the other's).

Overwrites the existing nhanes_model/grpo_decompv1/merged_model (previously
built from checkpoint-50) with checkpoint-239's weights — checkpoint-239
supersedes it on every eval_reward data point we have, so this is an
in-place upgrade, not a separate artifact.
"""
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SFT_MODEL = os.path.join(BASE_DIR, "nhanes_model", "decomp_v1", "merged_model")
# Training continued past checkpoint-50 for a full epoch (239 steps).
# checkpoint-239 is the final (most-trained) checkpoint, NOT the same as
# final_model/ — load_best_model_at_end picked final_model by eval_loss,
# which turned out to be checkpoint-50 itself (eval_loss doesn't track
# reward in this GRPO run; see training_history.json). eval_reward rose
# 0.228 (step 50) -> 0.411 (100) -> 0.473 (150) -> 0.471 (200); no held-out
# eval exists at 239 itself, only training-time reward (~0.46-0.55).
GRPO_LORA = os.path.join(BASE_DIR, "nhanes_model", "grpo_decompv1", "checkpoint-239")
OUTPUT_DIR = os.path.join(BASE_DIR, "nhanes_model", "grpo_decompv1", "merged_model")

print(f"Base model: {SFT_MODEL}")
print(f"GRPO LoRA:  {GRPO_LORA}")
print(f"Output:     {OUTPUT_DIR}")

print("\nLoading base model...")
model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL, trust_remote_code=True)

print("Loading GRPO LoRA adapter...")
model = PeftModel.from_pretrained(model, GRPO_LORA)

print("Merging...")
model = model.merge_and_unload()

print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

del model
gc.collect()
torch.cuda.empty_cache()
print("Done!")
