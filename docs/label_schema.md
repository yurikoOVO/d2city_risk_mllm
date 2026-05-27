# 标签体系

## Context 字段

| 字段 | 取值 | 说明 |
|---|---|---|
| weather | clear, rainy, foggy, night, cloudy | 天气或光照条件 |
| traffic_flow | low, medium, high | 交通流状态 |
| average_speed | integer | 平均速度，单位 km/h |
| event_info | none, nearby accident, road construction, large event ending nearby, vehicle breakdown | 外部事件信息 |

## 风险等级规则

1. `event_info != none` 且 `average_speed < 25`，标为 `high`。
2. `weather in rainy/foggy/night` 且 `traffic_flow == high`，标为 `high`。
3. `traffic_flow == high` 或 `average_speed < 35`，标为 `medium`。
4. 其他情况标为 `low`。

## 异常事件规则

| 条件 | abnormal_event |
|---|---|
| nearby accident | accident_related_congestion |
| road construction | construction_impact |
| large event ending nearby | event_related_congestion |
| vehicle breakdown | vehicle_breakdown |
| high flow and low speed | congestion |
| otherwise | normal |

## 中文输出格式

```text
风险等级：高风险。异常事件：事故相关拥堵。原因分析：...管理建议：...
```
