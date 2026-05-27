# D2-City Risk MLLM

本项目面向交通管理场景的“全路网视频巡检与异常事件自动发现”。输入为 D2-City 交通视频帧和外部结构化交通信息，输出安全风险等级、异常事件类型、原因解释和管理建议。

当前版本提供一套可运行的最小研究流水线：数据扫描、抽帧、弱标注 SFT 构造、数据切分、样本检查、Qwen2-VL/Qwen2.5-VL LoRA 微调、推理、base vs LoRA 对比和 Markdown 展示表生成。

## 1. 环境安装

```bash
cd /ssd/wyj/d2city_risk_mllm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如服务器已有 PyTorch/CUDA 环境，可先安装匹配 CUDA 的 `torch`，再安装其余依赖。

## 2. 数据扫描

默认数据集路径为 `/ssd/wyj/d2city`，配置在 `configs/default.yaml`。

```bash
python scripts/scan_d2city.py
```

输出：`data/raw_index.json`。

## 3. 抽帧

默认每隔 5 秒抽一帧；原始图片默认软链接到 `data/frames/`。

```bash
python scripts/extract_frames.py
```

输出：`data/frame_index.json` 和 `data/frames/`。

## 4. 构造 SFT

```bash
python scripts/build_risk_sft.py
```

输出：`data/risk_sft.jsonl`。每行包含 `image`、`context`、`question`、`answer`，并额外保留 `risk_level` 和 `abnormal_event` 便于统计分析。

`traffic_flow` 不再随机生成，而是基于视觉车辆密度估计：脚本使用 `ultralytics` YOLOv8n 检测 `car`、`bus`、`truck`、`motorcycle`，统计 `vehicle_count` 后按规则映射流量状态：`vehicle_count <= 3` 为 `low`，`4-8` 为 `medium`，`>8` 为 `high`。检测结果缓存到 `data/vehicle_count_cache.json`，后续重新构造样本会优先复用缓存，避免重复推理。`average_speed` 仍与 `traffic_flow` 绑定生成：`low` 为 50-80 km/h，`medium` 为 25-50 km/h，`high` 为 5-25 km/h。

## 5. 切分数据集

```bash
python scripts/split_dataset.py
```

默认比例：train/val/test = 0.8/0.1/0.1。

## 6. 检查样本

```bash
python scripts/check_samples.py --input data/risk_sft.jsonl --num_show 3 --check_images
```

## 7. LoRA 训练

```bash
python scripts/train_lora.py \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --train_jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl \
  --output_dir outputs/checkpoints/qwen_vl_lora \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_pixels 602112 \
  --save_steps 200 \
  --logging_steps 10 \
  --use_lora
```

训练脚本已实现最小可运行版本：使用 HuggingFace `Trainer`、`peft` LoRA、Qwen-VL chat template，并将用户输入部分的 labels mask 为 `-100`，只训练 assistant answer。后续可扩展点包括：packing、多卡 DeepSpeed、FlashAttention、量化加载、基于真实事件标签的监督信号和更细粒度评价指标。

## 8. 推理

只跑 base model：

```bash
python scripts/infer.py \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --lora_path "" \
  --image /absolute/path/to/frame.jpg \
  --weather rainy \
  --traffic_flow high \
  --average_speed 18 \
  --vehicle_count 12 \
  --event_info "nearby accident"
```

加载 LoRA：

```bash
python scripts/infer.py \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --lora_path outputs/checkpoints/qwen_vl_lora \
  --image /absolute/path/to/frame.jpg \
  --weather rainy \
  --traffic_flow high \
  --average_speed 18 \
  --vehicle_count 12 \
  --event_info "nearby accident"
```

## 9. Base vs LoRA 对比

```bash
python scripts/build_ablation_sets.py

python scripts/compare_base_lora.py \
  --base_model_path /ssd/wyj/models/Qwen2-VL-2B-Instruct \
  --finetuned_model_path outputs/checkpoints/qwen2vl_risk_lora \
  --test_jsonl data/test.jsonl \
  --num_samples 8 \
  --output outputs/infer_results.jsonl

python scripts/compare_base_lora.py \
  --base_model_path /ssd/wyj/models/Qwen2-VL-2B-Instruct \
  --finetuned_model_path outputs/checkpoints/qwen2vl_risk_lora \
  --num_samples 8 \
  --run_ablation

python scripts/evaluate_outputs.py --input outputs/ablation_full_context.jsonl

python scripts/results_to_markdown.py \
  --input outputs/infer_results.jsonl \
  --output outputs/compare_results.md
```

Markdown 表格可直接用于组会展示。

## 10. Git 版本管理方法

```bash
git status
git add README.md requirements.txt configs scripts docs
git commit -m "init d2city traffic risk mllm pipeline"
```

建议不要把 `data/frames/`、模型 checkpoint 和大 JSONL 直接提交到 Git；可以使用 `.gitignore` 或 Git LFS 管理。

## 推荐流水线

```bash
python scripts/scan_d2city.py
python scripts/extract_frames.py
python scripts/build_risk_sft.py
python scripts/split_dataset.py
python scripts/check_samples.py --input data/train.jsonl --num_show 2
```
