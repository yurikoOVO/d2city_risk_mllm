from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from common import PROJECT_ROOT, abs_path, load_config, read_json, write_json, write_jsonl


NON_NIGHT_WEATHERS = ["clear", "rainy", "foggy", "cloudy"]
EVENTS = ["none", "nearby accident", "road construction", "large event ending nearby", "vehicle breakdown"]
QUESTION = "请结合交通监控画面和外部交通信息，判断当前路段的安全风险等级、异常事件类型，并给出原因分析和管理建议。"
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

WEATHER_ZH = {"clear": "晴朗", "rainy": "雨天", "foggy": "雾天", "night": "夜间", "cloudy": "多云"}
FLOW_ZH = {"low": "低流量", "medium": "中等流量", "high": "高流量"}
WEATHER_DESC = {"clear": "晴朗天气", "rainy": "降雨天气", "foggy": "雾天低能见度", "night": "夜间运行环境", "cloudy": "多云天气"}
FLOW_DESC = {"low": "车流量较低", "medium": "车流量处于中等水平", "high": "车流量较高"}
EVENT_ZH = {
    "none": "无外部事件",
    "nearby accident": "附近事故",
    "road construction": "道路施工",
    "large event ending nearby": "附近大型活动散场",
    "vehicle breakdown": "车辆故障",
}
RISK_ZH = {"high": "高风险", "medium": "中风险", "low": "低风险"}
ABNORMAL_ZH = {
    "accident_related_congestion": "事故相关拥堵",
    "construction_impact": "施工影响",
    "event_related_congestion": "活动散场拥堵",
    "vehicle_breakdown": "车辆故障",
    "severe_congestion": "严重拥堵",
    "congestion": "交通拥堵",
    "normal": "正常通行",
}


def compute_brightness(image_path: str | Path) -> float | None:
    try:
        with Image.open(image_path) as image:
            gray = image.convert("L")
            brightness = ImageStat.Stat(gray).mean[0]
    except Exception as exc:
        print(f"[WARN] Failed to read image for brightness analysis: {image_path}, {exc}")
        return None
    return brightness


def infer_weather(text: str, rng: random.Random, brightness: float | None = None) -> str:
    if brightness is not None and brightness < 60:
        return "night"

    lower = text.lower()
    if any(k in lower for k in ["rain", "雨"]):
        return "rainy"
    if any(k in lower for k in ["fog", "雾"]):
        return "foggy"
    if any(k in lower for k in ["cloud", "阴", "多云"]):
        return "cloudy"
    return rng.choice(NON_NIGHT_WEATHERS)


class VehicleDensityEstimator:
    def __init__(self, cache_path: str | Path, model_name: str = "yolov8n.pt", conf: float = 0.25):
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.conf = conf
        self.model: Any | None = None
        self.cache: dict[str, int] = {}
        if self.cache_path.exists():
            cached = read_json(self.cache_path)
            self.cache = {str(k): int(v) for k, v in cached.items()}

    def _load_model(self) -> Any:
        if self.model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "Vehicle density estimation requires ultralytics. Install dependencies with: "
                    "pip install -r requirements.txt"
                ) from exc
            self.model = YOLO(self.model_name)
        return self.model

    def count_vehicles(self, image_path: str | Path) -> int:
        image_key = str(Path(image_path).resolve())
        if image_key in self.cache:
            return self.cache[image_key]

        model = self._load_model()
        results = model.predict(source=image_key, conf=self.conf, verbose=False)
        vehicle_count = 0
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for cls_id in boxes.cls.tolist():
                class_name = names.get(int(cls_id), "")
                if class_name in VEHICLE_CLASSES:
                    vehicle_count += 1

        self.cache[image_key] = vehicle_count
        self.save_cache()
        return vehicle_count

    def save_cache(self) -> None:
        write_json(self.cache, self.cache_path)


def infer_traffic_flow(vehicle_count: int) -> str:
    if vehicle_count <= 3:
        return "low"
    if vehicle_count <= 8:
        return "medium"
    return "high"


def sample_speed(flow: str, rng: random.Random) -> int:
    if flow == "high":
        return rng.randint(5, 25)
    if flow == "medium":
        return rng.randint(25, 50)
    return rng.randint(50, 80)


def decide_risk(weather: str, flow: str, speed: int, event_info: str) -> str:
    if event_info == "nearby accident":
        return "high"
    if event_info in {"road construction", "vehicle breakdown"}:
        return "medium"
    if event_info == "large event ending nearby" and flow in {"medium", "high"}:
        return "high"
    if event_info == "large event ending nearby":
        return "medium"
    if weather in {"rainy", "foggy", "night"} and flow == "high":
        return "high"
    if flow == "high" or speed < 35:
        return "medium"
    return "low"


def decide_abnormal(flow: str, speed: int, event_info: str) -> str:
    mapping = {
        "nearby accident": "accident_related_congestion",
        "road construction": "construction_impact",
        "large event ending nearby": "event_related_congestion",
        "vehicle breakdown": "vehicle_breakdown",
    }
    if event_info in mapping:
        return mapping[event_info]
    if flow == "high" and speed < 30:
        return "severe_congestion"
    if flow == "high":
        return "congestion"
    return "normal"


def make_answer(context: dict, risk_level: str, abnormal_event: str, rng: random.Random) -> str:
    weather = WEATHER_ZH[context["weather"]]
    weather_desc = WEATHER_DESC[context["weather"]]
    flow = FLOW_ZH[context["traffic_flow"]]
    flow_desc = FLOW_DESC[context["traffic_flow"]]
    speed = context["average_speed"]
    vehicle_count = context.get("vehicle_count")
    event = EVENT_ZH[context["event_info"]]
    risk = RISK_ZH[risk_level]
    abnormal = ABNORMAL_ZH[abnormal_event]

    if risk_level == "high":
        risk_tone = rng.choice(
            [
                "需纳入重点预警并尽快干预",
                "已经具备较明显的运行风险，应前置处置力量",
                "对路网通行稳定性影响较大，建议立即提升管控等级",
            ]
        )
        potential_risk = rng.choice(
            [
                "可能进一步诱发排队溢出、追尾或车辆滞留",
                "容易造成拥堵扩散，并增加次生事故概率",
                "若处置不及时，可能向相邻路段传导形成连锁拥堵",
            ]
        )
        advice_pool = [
            "加强视频巡检",
            "发布绕行提示",
            "启动分流",
            "增派值班人员",
            "联动交警",
            "关注次生事故风险",
        ]
    elif risk_level == "medium":
        risk_tone = rng.choice(
            [
                "存在一定运行风险，需要持续跟踪",
                "通行状态已有波动，应提高巡检频率",
                "短时内可能出现通行效率下降，建议提前关注",
            ]
        )
        potential_risk = rng.choice(
            [
                "后续若流量继续上升，可能形成局部拥堵",
                "受事件或环境影响，车辆变道和减速行为可能增多",
                "若速度继续下降，排队长度可能快速增加",
            ]
        )
        advice_pool = [
            "重点监测",
            "加强巡检",
            "限速提醒",
            "发布绕行提示",
            "关注次生事故风险",
        ]
    else:
        risk_tone = rng.choice(
            [
                "整体运行较为平稳",
                "当前路段通行状态总体可控",
                "路网运行未见明显异常波动",
            ]
        )
        potential_risk = rng.choice(
            [
                "短时风险主要来自流量突增或天气变化",
                "暂未发现明显拥堵扩散迹象",
                "仍需关注后续车速和车流变化",
            ]
        )
        advice_pool = ["保持常规巡检", "重点监测", "限速提醒"]

    advice_items = rng.sample(advice_pool, k=min(len(advice_pool), rng.choice([2, 3])))
    advice = "，".join(advice_items)

    event_sentence = (
        f"外部事件显示存在{event}"
        if context["event_info"] != "none"
        else "暂无明确外部突发事件"
    )
    abnormal_sentence = (
        f"研判为{abnormal}，"
        if abnormal_event != "normal"
        else "未识别到明确异常事件，"
    )
    reason_templates = [
        f"受{weather_desc}、{flow_desc}以及约{speed} km/h 的平均速度共同影响，画面估计车辆数约{vehicle_count}辆，{event_sentence}，{abnormal_sentence}{risk_tone}。",
        f"从路网运行状态看，{weather}条件下{flow}运行，画面约识别到{vehicle_count}辆机动车，车辆平均速度约{speed} km/h；同时{event_sentence}，{abnormal_sentence}{risk_tone}。",
        f"结合监控画面和外部信息，当前为{weather_desc}，路段呈现{flow_desc}，车辆密度估计约{vehicle_count}辆，平均速度约{speed} km/h，{event_sentence}，{risk_tone}。",
    ]
    reason = rng.choice(reason_templates)
    risk_sentence = f"{potential_risk}。"
    management = f"建议{advice}，并根据现场反馈动态调整管控措施。"
    if risk_level == "high":
        management = f"建议立即{advice}，必要时启动应急联动，并根据现场反馈动态调整管控措施。"
    else:
        management = f"建议{advice}，保持对车速、排队长度和相邻路段传导情况的跟踪。"

    return f"风险等级：{risk}。异常事件：{abnormal}。原因分析：{reason}{risk_sentence}管理建议：{management}"


def build_samples(
    frame_index: list[dict],
    seed: int,
    question: str,
    vehicle_estimator: VehicleDensityEstimator,
) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for item in frame_index:
        image = str(Path(item["frame_path"]).resolve())
        brightness = compute_brightness(image)
        weather = infer_weather(image + " " + item.get("scene_id", ""), rng, brightness)
        vehicle_count = vehicle_estimator.count_vehicles(image)
        flow = infer_traffic_flow(vehicle_count)
        speed = sample_speed(flow, rng)
        event_info = rng.choice(EVENTS)
        context = {
            "weather": weather,
            "traffic_flow": flow,
            "average_speed": speed,
            "vehicle_count": vehicle_count,
            "event_info": event_info,
        }
        risk_level = decide_risk(weather, flow, speed, event_info)
        abnormal_event = decide_abnormal(flow, speed, event_info)
        samples.append(
            {
                "image": image,
                "context": context,
                "question": question,
                "answer": make_answer(context, risk_level, abnormal_event, rng),
                "risk_level": risk_level,
                "abnormal_event": abnormal_event,
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weakly labeled multimodal traffic risk SFT samples.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--frame_index", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vehicle_cache", default=None)
    parser.add_argument("--yolo_model", default=None)
    parser.add_argument("--yolo_conf", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    frame_index = read_json(abs_path(args.frame_index or cfg["data"]["frame_index"]))
    question = cfg.get("sft", {}).get("question", QUESTION)
    seed = args.seed if args.seed is not None else cfg.get("sft", {}).get("seed", 42)
    vehicle_estimator = VehicleDensityEstimator(
        cache_path=abs_path(args.vehicle_cache or cfg["data"].get("vehicle_cache", "data/vehicle_count_cache.json")),
        model_name=args.yolo_model or cfg.get("sft", {}).get("yolo_model", "yolov8n.pt"),
        conf=args.yolo_conf if args.yolo_conf is not None else cfg.get("sft", {}).get("yolo_conf", 0.25),
    )
    samples = build_samples(frame_index, seed, question, vehicle_estimator)
    vehicle_estimator.save_cache()
    output = abs_path(args.output or cfg["data"]["sft_jsonl"])
    write_jsonl(samples, output)
    print(f"Saved {len(samples)} SFT samples to {output}")


if __name__ == "__main__":
    main()
