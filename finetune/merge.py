"""Merge the LoRA adapter into the bf16 base model so vLLM can serve it.

  python merge.py --adapter t5_lora --output-dir qwen3.5-27b-t5-merged
  vllm serve qwen3.5-27b-t5-merged --port 8000    # then evaluate with ../harness (T5, Non-Thinking)
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--adapter", required=True, help="output directory of train.py")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                       device_map="auto", trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
