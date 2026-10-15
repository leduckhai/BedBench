"""QLoRA fine-tuning of Qwen3.5-27B on T5 (Color Distribution), as in the paper's fine-tuning study.

  python train.py --dataset bed_dataset/T5_finetune --output-dir t5_lora            # model split over all GPUs
  accelerate launch --num_processes 4 train.py --ddp --grad-accum 2 \
      --dataset bed_dataset/T5_finetune --output-dir t5_lora                         # one model copy per GPU

The dataset is written by `python -m bed_generator.build_tasks ... --finetune-samples POOL` (train: 800
objects, validation: 100). Each example is the two default color views, the Non-Thinking T5 prompt, and the
target JSON {"answer": {color: count, ...}}. The checkpoint with the lowest validation loss is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, TaskType
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "non_thinking" / "Task_5.md"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def messages(example: dict, instruction: str) -> list[dict]:
    target = {"answer": json.loads(example["color_distribution_json"])}
    return [{"role": "user", "content": [{"type": "image", "image": example["query_image_1"]},
                                         {"type": "image", "image": example["query_image_2"]},
                                         {"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target)}]}]


def collate(examples: list[dict], processor, instruction: str, max_length: int) -> dict:
    """Chat-formatted batch. Labels are the input ids, so the loss covers the whole sequence (prompt and
    answer), as in the paper's run."""
    texts = [processor.apply_chat_template(messages(ex, instruction), tokenize=False, add_generation_prompt=False)
             for ex in examples]
    images = [[ex["query_image_1"], ex["query_image_2"]] for ex in examples]
    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt")
    batch["labels"] = batch["input_ids"].clone()
    return batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="save_to_disk directory with train and validation splits, or the BedBench dataset repo id")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--eval-steps", type=int, default=50, help="validation and checkpoint interval")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke tests)")
    ap.add_argument("--ddp", action="store_true",
                    help="data parallel under accelerate launch; keep batch-size x grad-accum x processes = 8")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --output-dir")
    args = ap.parse_args()

    if args.ddp:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dataset = (load_from_disk(args.dataset) if os.path.isdir(args.dataset)
               else load_dataset(args.dataset, "T5_finetune"))  # local save_to_disk dir or the HF dataset repo
    instruction = PROMPT.read_text(encoding="utf-8").strip()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))} if args.ddp else "auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True))
    model.config.use_cache = False  # required for gradient checkpointing

    config = SFTConfig(
        output_dir=args.output_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.1, weight_decay=0.01, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, logging_steps=5,
        eval_strategy="steps", eval_steps=args.eval_steps, save_steps=args.eval_steps,
        # DDP: PEFT reloads the best adapter onto "cuda" (GPU 0) in every process and runs out of memory, so the
        # best checkpoint is copied after training instead; keep all checkpoints so it cannot be rotated away.
        save_total_limit=None if args.ddp else 3, load_best_model_at_end=not args.ddp,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", max_length=args.max_seq_length, dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        ddp_find_unused_parameters=False if args.ddp else None)
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        processing_class=processor,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                               target_modules=LORA_TARGETS, task_type=TaskType.CAUSAL_LM, bias="none"),
        data_collator=lambda batch: collate(batch, processor, instruction, args.max_seq_length))
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    best = trainer.state.best_model_checkpoint
    if not args.ddp:
        trainer.save_model(args.output_dir)   # LoRA adapter of the best checkpoint (reloaded by the Trainer)
    elif trainer.is_world_process_zero():
        if best is None:
            raise RuntimeError("no evaluated checkpoint to select the best adapter from")
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copyfile(Path(best) / name, Path(args.output_dir) / name)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir)
        print(f"best checkpoint (lowest eval_loss): {best}")


if __name__ == "__main__":
    main()
