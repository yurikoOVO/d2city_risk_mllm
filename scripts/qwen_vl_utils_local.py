from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_QUESTION = "请结合交通监控画面和外部交通信息，判断当前路段的安全风险等级、异常事件类型，并给出原因分析和管理建议。"


def format_context(context: dict[str, Any]) -> str:
    if not context:
        return "外部结构化交通信息：未提供。\n"

    lines = ["外部结构化交通信息："]
    if "weather" in context:
        lines.append(f"- weather: {context['weather']}")
    if "traffic_flow" in context:
        lines.append(f"- traffic_flow: {context['traffic_flow']}")
    if "average_speed" in context:
        lines.append(f"- average_speed: {context['average_speed']} km/h")
    if "vehicle_count" in context:
        lines.append(f"- vehicle_count: {context['vehicle_count']}")
    if "event_info" in context:
        lines.append(f"- event_info: {context['event_info']}")
    else:
        lines.append("- event_info: 未提供")
    return "\n".join(lines) + "\n"


def build_user_text(question: str, context: dict[str, Any]) -> str:
    return f"{question}\n\n{format_context(context)}"


def build_messages(
    image_path: str | Path,
    question: str,
    context: dict[str, Any],
    answer: str | None = None,
    max_pixels: int | None = None,
) -> list[dict[str, Any]]:
    image_content: dict[str, Any] = {"type": "image", "image": str(image_path)}
    if max_pixels is not None:
        image_content["max_pixels"] = max_pixels
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": build_user_text(question, context)},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def load_qwen_vl_model(model_path: str, torch_dtype: Any = "auto"):
    from transformers import Qwen2VLForConditionalGeneration

    return Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
    )


def process_vision_info_compat(messages: list[dict[str, Any]]):
    try:
        from qwen_vl_utils import process_vision_info

        return process_vision_info(messages)
    except ImportError:
        image_inputs = []
        video_inputs = []
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    image_inputs.append(Image.open(item["image"]).convert("RGB"))
                elif item.get("type") == "video":
                    raise ImportError(
                        "Video inputs require qwen-vl-utils. Install it with: pip install qwen-vl-utils"
                    )
        return image_inputs or None, video_inputs or None
