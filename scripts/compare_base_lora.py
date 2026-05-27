from __future__ import annotations

import argparse
import gc
import random
from typing import Any

import torch
from transformers import AutoProcessor

from common import abs_path, load_config, read_jsonl, write_jsonl
from qwen_vl_utils_local import build_messages, load_qwen_vl_model, process_vision_info_compat


ABLATION_JOBS = [
    ("image_only", "data/test_image_only.jsonl", "outputs/ablation_image_only.jsonl"),
    ("basic_context", "data/test_basic_context.jsonl", "outputs/ablation_basic_context.jsonl"),
    ("full_context", "data/test_full_context.jsonl", "outputs/ablation_full_context.jsonl"),
]


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


def load_model_and_processor(model_path: str, label: str) -> tuple[Any, Any]:
    print(f"Loading {label} model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = load_qwen_vl_model(model_path)
    model.eval()
    return model, processor


def unload_model_and_processor(model: Any, processor: Any) -> None:
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_loaded_model_on_samples(
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
    max_pixels: int,
    label: str,
) -> list[str]:
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
    return outputs


def sample_rows(test_jsonl: str, num_samples: int, seed: int) -> list[dict[str, Any]]:
    rows = read_jsonl(test_jsonl)
    rng = random.Random(seed)
    return rng.sample(rows, min(num_samples, len(rows)))

def build_output_rows(
    samples: list[dict[str, Any]],
    base_outputs: list[str],
    finetuned_outputs: list[str],
) -> list[dict[str, Any]]:
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
    return outputs


def make_jobs(args: argparse.Namespace) -> list[tuple[str, list[dict[str, Any]], str]]:
    if args.run_ablation:
        return [
            (job_name, sample_rows(str(abs_path(test_jsonl)), args.num_samples, args.seed), str(abs_path(output_path)))
            for job_name, test_jsonl, output_path in ABLATION_JOBS
        ]
    return [("default", sample_rows(args.test_jsonl, args.num_samples, args.seed), args.output)]


def run_jobs(args: argparse.Namespace, jobs: list[tuple[str, list[dict[str, Any]], str]]) -> None:
    base_model, base_processor = load_model_and_processor(args.base_model_path, "base")
    try:
        base_outputs_by_job = {
            name: run_loaded_model_on_samples(
                model=base_model,
                processor=base_processor,
                samples=samples,
                max_new_tokens=args.max_new_tokens,
                max_pixels=args.max_pixels,
                label=f"{name} base",
            )
            for name, samples, _ in jobs
        }
    finally:
        unload_model_and_processor(base_model, base_processor)

    finetuned_model, finetuned_processor = load_model_and_processor(args.finetuned_model_path, "finetuned")
    try:
        for name, samples, output_path in jobs:
            finetuned_outputs = run_loaded_model_on_samples(
                model=finetuned_model,
                processor=finetuned_processor,
                samples=samples,
                max_new_tokens=args.max_new_tokens,
                max_pixels=args.max_pixels,
                label=f"{name} finetuned",
            )
            outputs = build_output_rows(samples, base_outputs_by_job[name], finetuned_outputs)
            write_jsonl(outputs, output_path)
            print(f"Saved comparison results to {output_path}")
    finally:
        unload_model_and_processor(finetuned_model, finetuned_processor)


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
    parser.add_argument("--run_ablation", action="store_true", help="Run image-only/basic/full context ablation sets.")
    args = parser.parse_args()

    run_jobs(args, make_jobs(args))


if __name__ == "__main__":
    main()
