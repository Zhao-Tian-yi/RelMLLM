#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
from tools.qwen3_aligned_rope_patch import apply_qwen3_vl_aligned_pair_rope_patch


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class RelationSFTDataset(Dataset):
    def __init__(self, path: str | Path):
        self.rows = load_jsonl(Path(path))
        if not self.rows:
            raise ValueError("Training/evaluation JSONL must not be empty")
        root = Path(path).resolve().parent
        for index, row in enumerate(self.rows):
            messages = row.get("messages", [])
            if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"Row {index}: expected system/user/assistant messages")
            if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
                raise ValueError(f"Row {index}: message contents must be nonempty strings")
            if len(row.get("images", [])) != 2:
                raise ValueError(f"Row {index}: expected RGB then TIR image paths")
            paths = []
            for value in row["images"]:
                image_path = Path(value)
                if not image_path.is_absolute():
                    image_path = root / image_path
                with Image.open(image_path) as image:
                    image.verify()
                paths.append(str(image_path))
            row["images"] = paths

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class RelationSFTCollator:
    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length
        if processor.tokenizer.padding_side != "right":
            raise ValueError("SFT loss masking requires right padding")

    def _messages_with_images(self, row: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
        messages = []
        rgb_path, tir_path = row["images"]
        source_messages = row["messages"] if include_answer else row["messages"][:2]
        for msg in source_messages:
            if msg["role"] == "user":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": load_rgb(rgb_path)},
                            {"type": "image", "image": load_rgb(tir_path)},
                            {"type": "text", "text": msg["content"]},
                        ],
                    }
                )
            else:
                messages.append({"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]})
        return messages

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_inputs = []
        prompt_lengths = []
        for row in features:
            full_messages = self._messages_with_images(row, include_answer=True)
            prompt_messages = self._messages_with_images(row, include_answer=False)
            full = self.processor.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"truncation": False},
            )
            prompt = self.processor.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"truncation": False},
            )
            if full["input_ids"].shape[-1] > self.max_length:
                raise ValueError("Sequence exceeds --max_length; increase it or reduce image/text size")
            prompt_len = prompt["input_ids"].shape[-1]
            if not torch.equal(full["input_ids"][:, :prompt_len], prompt["input_ids"]):
                raise ValueError("Chat-template prompt is not a prefix of the training conversation")
            batch_inputs.append(full)
            prompt_lengths.append(prompt_len)

        sequence_keys = {"input_ids", "attention_mask"}
        padded = self.processor.tokenizer.pad(
            [{k: v.squeeze(0) for k, v in item.items() if k in sequence_keys} for item in batch_inputs],
            padding=True,
            return_tensors="pt",
        )
        if any("mm_token_type_ids" in item for item in batch_inputs):
            max_len = padded["input_ids"].shape[1]
            mm_rows = []
            for item in batch_inputs:
                if "mm_token_type_ids" not in item:
                    row = torch.zeros(max_len, dtype=torch.long)
                else:
                    row = item["mm_token_type_ids"].squeeze(0).to(dtype=torch.long)
                    pad_len = max_len - row.shape[0]
                    if pad_len < 0:
                        row = row[-max_len:] if self.processor.tokenizer.padding_side == "left" else row[:max_len]
                        pad_len = 0
                    pad = torch.zeros(pad_len, dtype=row.dtype)
                    row = torch.cat([pad, row], dim=0) if self.processor.tokenizer.padding_side == "left" else torch.cat([row, pad], dim=0)
                mm_rows.append(row)
            padded["mm_token_type_ids"] = torch.stack(mm_rows, dim=0)
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, : min(prompt_len, labels.shape[1])] = -100
            if not (labels[i] != -100).any():
                raise ValueError("No assistant tokens remain; increase --max_length")

        # Keep non-text multimodal tensors from processor output when batch size is one.
        # Qwen-VL processors differ across versions; use processor.pad-compatible text tensors
        # plus concatenate vision tensors when present.
        out = dict(padded)
        out["labels"] = labels
        extra_keys = set().union(*(set(item.keys()) for item in batch_inputs)) - (sequence_keys | {"mm_token_type_ids"})
        for key in extra_keys:
            values = [item[key] for item in batch_inputs if key in item]
            if values and torch.is_tensor(values[0]):
                try:
                    out[key] = torch.cat(values, dim=0)
                except RuntimeError as error:
                    raise ValueError(f"Cannot concatenate multimodal tensor {key}") from error
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for RGBT relation seed schema alignment.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_path", default=None, help="Optional trainable LoRA adapter for continual SFT.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge_output_dir", help="Optionally also export merged inference weights")
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--save_steps", type=int, default=40)
    parser.add_argument("--eval_steps", type=int, default=40)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length < 1:
        raise ValueError("--max_length must be positive")
    if args.merge_output_dir and int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("--merge_output_dir requires single-process training")
    for value in (args.output_dir, args.merge_output_dir):
        if value and Path(value).exists() and any(Path(value).iterdir()):
            if value != args.output_dir or not args.resume_from_checkpoint:
                raise FileExistsError(f"Refusing to overwrite nonempty output directory: {value}")
    train_dataset = RelationSFTDataset(args.train_jsonl)
    eval_dataset = RelationSFTDataset(args.eval_jsonl)
    print(f"[RelationSFT] train={len(train_dataset)} eval={len(eval_dataset)}")
    if args.dry_run:
        print("[RelationSFT] dry_run complete; model not loaded.")
        return

    apply_qwen3_vl_aligned_pair_rope_patch()
    from transformers import set_seed
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.enable_input_require_grads()
    if args.adapter_path:
        print(f"[RelationSFT] loading trainable adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RelationSFTCollator(processor, max_length=args.max_length),
    )
    resume_from_checkpoint: str | bool | None = args.resume_from_checkpoint
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == "true":
        resume_from_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir + "/final_model")
    processor.save_pretrained(args.output_dir + "/final_model")
    if args.merge_output_dir:
        if training_args.world_size != 1:
            raise ValueError("Merge export requires single-process training; merge adapters separately for distributed runs")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merge_output_dir, safe_serialization=True)
        processor.save_pretrained(args.merge_output_dir)


if __name__ == "__main__":
    main()
