# 输出 JSON Schema（统一格式）

更新时间：2026-01-23

## 1. Schema（JSON Schema Draft 7）
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RCAResult",
  "type": "object",
  "required": ["event_id", "time", "top_candidates", "prediction"],
  "properties": {
    "event_id": { "type": "string" },
    "time": { "type": "string" },
    "top_candidates": {
      "type": "array",
      "items": { "type": "string" }
    },
    "prediction": {
      "type": "object",
      "required": ["root_cause_component", "fault_type", "kpi", "related_container"],
      "properties": {
        "root_cause_component": { "type": "string" },
        "fault_type": { "type": "string" },
        "kpi": { "type": "string" },
        "related_container": { "type": "string" },
        "explanation": { "type": "string" }
      }
    }
  }
}
```

## 2. 示例
```json
{
  "event_id": "1",
  "time": "2020-04-11 00:05:00",
  "top_candidates": ["docker_003", "docker_002", "docker_001"],
  "prediction": {
    "root_cause_component": "docker_003",
    "fault_type": "CPU fault",
    "kpi": "container_cpu_used",
    "related_container": "container_001",
    "explanation": "证据显示 container_cpu_used 在窗口内显著升高，且调用链延迟集中在 docker_003。"
  }
}
```
