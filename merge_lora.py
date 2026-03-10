"""
merge_lora.py - Merge LoRA adapter into base model
====================================================
Merges LoRA weights into base model so sglang can load it directly.

Usage:
    python merge_lora.py --exp 1                  # merge one
    python merge_lora.py --exp 1 2 3 4 5          # merge all
    python merge_lora.py --exp 2 --model_name /path/to/base
"""

import os
import gc
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "nhanes_model")
MODEL_NAME = "Qwen/Qwen3-8B-Base"


def merge_one(exp_id, model_name):
    lora_dir = os.path.join(OUTPUT_ROOT, f"exp{exp_id}", "best_model")
    merged_dir = os.path.join(OUTPUT_ROOT, f"exp{exp_id}", "merged_model")

    if not os.path.exists(lora_dir):
        print(f"  ERROR: LoRA not found at {lora_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  Merging Exp{exp_id}: {lora_dir}")
    print(f"  Output: {merged_dir}")
    print(f"{'='*60}")

    # Load base model
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load base tokenizer (NOT the LoRA one, to avoid vocab mismatch)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )

    # Load and merge LoRA
    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    print("  Merging...")
    model = model.merge_and_unload()

    # Save merged model
    print("  Saving merged model...")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"  Done! Saved to {merged_dir}")

    # Cleanup
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    print("  GPU memory cleared.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--exp", type=int, nargs="+", required=True)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    for exp_id in args.exp:
        merge_one(exp_id, args.model_name)

    print(f"\nAll done!")


if __name__ == "__main__":
    main()