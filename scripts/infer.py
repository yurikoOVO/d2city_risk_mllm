from __future__ import annotations

import argparse
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor

from qwen_vl_utils_local import DEFAULT_QUESTION, build_messages, load_qwen_vl_model, process_vision_info_compat


def generate_risk_analysis(
    model_path: str,
    lora_path: str,
    image: str,
    context: dict[str, Any],
    question: str = DEFAULT_QUESTION,
    max_new_tokens: int = 512,
    max_pixels: int | None = None,
) -> str:
    processor_path = lora_path if lora_path else model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model = load_qwen_vl_model(model_path)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    messages = build_messages(image, question, context, None, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info_compat(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen-VL traffic risk inference.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default="")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--traffic_flow", required=True, choices=["low", "medium", "high"])
    parser.add_argument("--average_speed", required=True, type=int)
    parser.add_argument("--vehicle_count", type=int, required=True, help="Estimated vehicle count")
    parser.add_argument("--event_info", required=True)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_pixels", type=int, default=602112)
    args = parser.parse_args()

    context = {
        "weather": args.weather,
        "traffic_flow": args.traffic_flow,
        "average_speed": args.average_speed,
        "vehicle_count": args.vehicle_count,
        "event_info": args.event_info,
    }
    print(
        generate_risk_analysis(
            args.model_path,
            args.lora_path,
            args.image,
            context,
            args.question,
            args.max_new_tokens,
            args.max_pixels,
        )
    )


if __name__ == "__main__":
    main()
