from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainingArguments

from common import PROJECT_ROOT, abs_path, load_config, read_jsonl
from qwen_vl_utils_local import build_messages, load_qwen_vl_model, process_vision_info_compat


IGNORE_INDEX = -100
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
REQUIRED_PROCESSOR_KEYS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
OPTIONAL_SEQUENCE_KEYS = ["mm_token_type_ids"]


class TrafficRiskDataset(Dataset):
    def __init__(self, jsonl_path: str | Path):
        self.rows = read_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


@dataclass
class QwenVLCollator:
    processor: Any
    max_pixels: int

    def _validate_processor_output(self, encoded: dict[str, torch.Tensor]) -> None:
        missing = REQUIRED_PROCESSOR_KEYS - set(encoded)
        if missing:
            available = ", ".join(sorted(encoded.keys()))
            raise ValueError(
                "Qwen2-VL processor output is missing required keys: "
                f"{sorted(missing)}. Available keys: {available}. "
                "Please make sure AutoProcessor is loaded from the same Qwen2-VL model path."
            )

    def _encode_one(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        full_messages = build_messages(sample["image"], sample["question"], sample["context"], sample["answer"], self.max_pixels)
        prompt_messages = build_messages(sample["image"], sample["question"], sample["context"], None, self.max_pixels)

        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info_compat(full_messages)
        full = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        self._validate_processor_output(full)
        prompt = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        if "input_ids" not in prompt:
            raise ValueError("Qwen2-VL processor prompt output is missing input_ids.")

        keep_batch_keys = {"pixel_values", "image_grid_thw", "video_grid_thw"}
        item = {k: v.squeeze(0) if torch.is_tensor(v) and v.dim() > 1 and k not in keep_batch_keys else v for k, v in full.items()}
        input_ids = item["input_ids"]
        labels = input_ids.clone()
        prompt_len = min(prompt["input_ids"].shape[-1], labels.shape[-1])
        labels[:prompt_len] = IGNORE_INDEX
        item["labels"] = labels
        return item

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_one(sample) for sample in features]
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(x["input_ids"].shape[0] for x in encoded)

        batch: dict[str, list[torch.Tensor] | torch.Tensor] = {"input_ids": [], "attention_mask": [], "labels": []}
        for key in OPTIONAL_SEQUENCE_KEYS:
            if key in encoded[0]:
                batch[key] = []
        for item in encoded:
            pad_len = max_len - item["input_ids"].shape[0]
            batch["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_id))
            batch["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
            batch["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX))
            for key in OPTIONAL_SEQUENCE_KEYS:
                if key in batch:
                    batch[key].append(torch.nn.functional.pad(item[key], (0, pad_len), value=0))

        out = {k: torch.stack(v) for k, v in batch.items()}  # type: ignore[arg-type]
        if "pixel_values" in encoded[0]:
            out["pixel_values"] = torch.cat([x["pixel_values"] for x in encoded], dim=0)
        if "image_grid_thw" in encoded[0]:
            out["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in encoded], dim=0)
        if "video_grid_thw" in encoded[0]:
            out["video_grid_thw"] = torch.cat([x["video_grid_thw"] for x in encoded], dim=0)
        return out


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    train_cfg = cfg["training"]
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2-VL/Qwen2.5-VL traffic risk SFT.")
    parser.add_argument("--model_path", default=train_cfg["model_path"])
    parser.add_argument("--train_jsonl", default=str(abs_path(cfg["data"]["train_jsonl"])))
    parser.add_argument("--val_jsonl", default=str(abs_path(cfg["data"]["val_jsonl"])))
    parser.add_argument("--output_dir", default=str(abs_path(train_cfg["output_dir"])))
    parser.add_argument("--num_train_epochs", type=float, default=train_cfg["num_train_epochs"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=train_cfg["per_device_train_batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=train_cfg["gradient_accumulation_steps"])
    parser.add_argument("--learning_rate", type=float, default=train_cfg["learning_rate"])
    parser.add_argument("--max_pixels", type=int, default=train_cfg["max_pixels"])
    parser.add_argument("--save_steps", type=int, default=train_cfg["save_steps"])
    parser.add_argument("--logging_steps", type=int, default=train_cfg["logging_steps"])
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=train_cfg.get("gradient_checkpointing", True))
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_qwen_vl_model(args.model_path)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = TrafficRiskDataset(args.train_jsonl)
    val_dataset = TrafficRiskDataset(args.val_jsonl) if Path(args.val_jsonl).exists() else None
    collator = QwenVLCollator(processor=processor, max_pixels=args.max_pixels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if val_dataset and len(val_dataset) > 0 else "no",
        eval_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved model/adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
