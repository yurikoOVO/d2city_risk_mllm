from __future__ import annotations

import argparse
import gc
import random
from typing import Any

import torch
from transformers import AutoProcessor

from common import abs_path, load_config, read_jsonl, write_jsonl
from qwen_vl_utils_local import build_messages, load_qwen_vl_model, process_vision_info_compat


def generate_with_loaded_model(
    model: Any,
    processor: Any,
    image: str,
    context: dict[str, Any],
    question: str,
    max_new_tokens: int,
    max_pixels: int,
) -> str:
    messages = build_messages(image, question, context, None, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info_compat(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def run_model_on_samples(
    model_path: str,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
    max_pixels: int,
    label: str,
) -> list[str]:
    print(f"Loading {label} model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = load_qwen_vl_model(model_path)
    model.eval()

    outputs = []
    for idx, sample in enumerate(samples, start=1):
        print(f"[{label} {idx}/{len(samples)}] {sample['image']}")
        outputs.append(
            generate_with_loaded_model(
                model=model,
                processor=processor,
                image=sample["image"],
                context=sample["context"],
                question=sample["question"],
                max_new_tokens=max_new_tokens,
                max_pixels=max_pixels,
            )
        )

    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Compare base Qwen2-VL and full fine-tuned Qwen2-VL outputs.")
    parser.add_argument("--base_model_path", default=cfg["training"]["model_path"])
    parser.add_argument("--finetuned_model_path", required=True)
    parser.add_argument("--test_jsonl", default=str(abs_path(cfg["data"]["test_jsonl"])))
    parser.add_argument("--output", default=str(abs_path("outputs/infer_results.jsonl")))
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_pixels", type=int, default=cfg["training"].get("max_pixels", 602112))
    args = parser.parse_args()

    rows = read_jsonl(args.test_jsonl)
    rng = random.Random(args.seed)
    samples = rng.sample(rows, min(args.num_samples, len(rows)))

    base_outputs = run_model_on_samples(
        model_path=args.base_model_path,
        samples=samples,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        label="base",
    )
    finetuned_outputs = run_model_on_samples(
        model_path=args.finetuned_model_path,
        samples=samples,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        label="finetuned",
    )

    outputs = []
    for sample, base_output, finetuned_output in zip(samples, base_outputs, finetuned_outputs):
        outputs.append(
            {
                "image": sample["image"],
                "context": sample["context"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "base_output": base_output,
                "finetuned_output": finetuned_output,
                "lora_output": finetuned_output,
            }
        )

    write_jsonl(outputs, args.output)
    print(f"Saved comparison results to {args.output}")


if __name__ == "__main__":
    main()
