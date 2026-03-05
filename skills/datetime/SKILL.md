---
name: datetime
description: 查询当前日期、时间、星期几等本地时间信息
params_hint: '{"skill": "datetime", "query_type": "full"|"date"|"time"|"weekday"}'
---

# 日期时间查询技能

查询运行环境的当前本地日期和时间信息。

## 适用场景

- 用户询问"今天是几号"、"今天是几月几日"
- 用户询问"现在几点了"、"当前时间是多少"
- 用户询问"今天星期几"、"今天是周几"
- 用户询问"现在的完整日期时间"

## 参数说明

| 参数        | 类型   | 可选值                              | 默认值 |
|-------------|--------|-------------------------------------|--------|
| query_type  | string | full \| date \| time \| weekday     | full   |

- `full`    — 完整日期 + 时间 + 星期（如：2025年06月01日 14:30:00 星期日）
- `date`    — 仅年月日（如：2025年06月01日）
- `time`    — 仅时分秒（如：14:30:00）
- `weekday` — 仅星期几（如：星期日）

## 示例

```json
// 查询完整日期时间
{"skill": "datetime", "query_type": "full"}

// 只查询日期
{"skill": "datetime", "query_type": "date"}

// 只查询时间
{"skill": "datetime", "query_type": "time"}

// 只查询星期几
{"skill": "datetime", "query_type": "weekday"}
```

## 返回值

格式化的中文日期/时间字符串。
