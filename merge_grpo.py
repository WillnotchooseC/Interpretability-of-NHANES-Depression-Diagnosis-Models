"""
merge_grpo.py - Merge GRPO LoRA adapter into SFT merged model
"""
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# GRPO was trained on top of exp1's merged model
SFT_MODEL = os.path.join(BASE_DIR, "nhanes_model", "exp1", "merged_model")
GRPO_LORA = os.path.join(BASE_DIR, "nhanes_model", "grpo", "final_model")
OUTPUT_DIR = os.path.join(BASE_DIR, "nhanes_model", "grpo", "merged_model")

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