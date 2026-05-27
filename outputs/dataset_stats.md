# Dataset Statistics

- Total samples: 603

## Average Speed

| min | max | mean |
|---:|---:|---:|
| 5.00 | 80.00 | 43.47 |

## Vehicle Count

| min | max | mean |
|---:|---:|---:|
| 0.00 | 18.00 | 5.18 |

## Risk Level Distribution

| value | count | ratio |
|---|---:|---:|
| medium | 332 | 55.06% |
| high | 204 | 33.83% |
| low | 67 | 11.11% |

## Abnormal Event Distribution

| value | count | ratio |
|---|---:|---:|
| vehicle_breakdown | 136 | 22.55% |
| event_related_congestion | 122 | 20.23% |
| accident_related_congestion | 121 | 20.07% |
| construction_impact | 106 | 17.58% |
| normal | 93 | 15.42% |
| severe_congestion | 16 | 2.65% |
| congestion | 9 | 1.49% |

## Weather Distribution

| value | count | ratio |
|---|---:|---:|
| rainy | 147 | 24.38% |
| foggy | 139 | 23.05% |
| cloudy | 135 | 22.39% |
| clear | 119 | 19.73% |
| night | 63 | 10.45% |

## Traffic Flow Distribution

| value | count | ratio |
|---|---:|---:|
| medium | 269 | 44.61% |
| low | 222 | 36.82% |
| high | 112 | 18.57% |

## Event Info Distribution

| value | count | ratio |
|---|---:|---:|
| vehicle breakdown | 137 | 22.72% |
| large event ending nearby | 132 | 21.89% |
| nearby accident | 121 | 20.07% |
| road construction | 110 | 18.24% |
| none | 103 | 17.08% |

## Risk Level x Abnormal Event

| risk_level | accident_related_congestion | congestion | construction_impact | event_related_congestion | normal | severe_congestion | vehicle_breakdown | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 0 | 0 | 5 | 0 | 62 | 0 | 0 | 67 |
| medium | 10 | 6 | 101 | 40 | 31 | 8 | 136 | 332 |
| high | 111 | 3 | 0 | 82 | 0 | 8 | 0 | 204 |

## Conditional Probability: Event Info -> Abnormal Event

| event_info | total | accident_related_congestion | congestion | construction_impact | event_related_congestion | normal | severe_congestion | vehicle_breakdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large event ending nearby | 132 | 0.00% | 0.00% | 0.00% | 92.42% | 7.58% | 0.00% | 0.00% |
| nearby accident | 121 | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| none | 103 | 0.00% | 8.74% | 0.00% | 0.00% | 75.73% | 15.53% | 0.00% |
| road construction | 110 | 0.00% | 0.00% | 96.36% | 0.00% | 3.64% | 0.00% | 0.00% |
| vehicle breakdown | 137 | 0.00% | 0.00% | 0.00% | 0.00% | 0.73% | 0.00% | 99.27% |

