"""
单位换算技能 — 执行脚本。

入口函数：execute(value, from_unit, to_unit) -> str

支持类别：
  长度  — km / mile / m / ft / cm / inch（及中文别名）
  重量  — kg / lb / g / oz（及中文别名）
  温度  — celsius / fahrenheit（及中文别名）
"""

from typing import Callable

# ── 别名归一化表（中英文 → 标准键）──────────────────────────────
_ALIASES: dict[str, str] = {
    # 长度
    "千米": "km",   "公里": "km",
    "英里": "mile",
    "米":   "m",
    "英尺": "ft",   "尺": "ft",
    "厘米": "cm",   "公分": "cm",
    "英寸": "inch", "寸": "inch",
    # 重量
    "千克": "kg",   "公斤": "kg",
    "磅":   "lb",
    "克":   "g",
    "盎司": "oz",
    # 温度
    "摄氏度": "celsius",    "℃": "celsius",    "摄氏": "celsius",
    "华氏度": "fahrenheit", "℉": "fahrenheit", "华氏": "fahrenheit",
}

# ── 换算函数表：(from_key, to_key) → Callable[[float], float] ───
_CONVERSIONS: dict[tuple[str, str], Callable[[float], float]] = {
    # 长度
    ("km",   "mile"): lambda x: x * 0.621_371,
    ("mile", "km"):   lambda x: x * 1.609_34,
    ("m",    "ft"):   lambda x: x * 3.280_84,
    ("ft",   "m"):    lambda x: x / 3.280_84,
    ("cm",   "inch"): lambda x: x / 2.54,
    ("inch", "cm"):   lambda x: x * 2.54,
    ("km",   "m"):    lambda x: x * 1_000,
    ("m",    "km"):   lambda x: x / 1_000,
    ("m",    "cm"):   lambda x: x * 100,
    ("cm",   "m"):    lambda x: x / 100,
    ("ft",   "inch"): lambda x: x * 12,
    ("inch", "ft"):   lambda x: x / 12,
    # 重量
    ("kg",   "lb"):   lambda x: x * 2.204_62,
    ("lb",   "kg"):   lambda x: x / 2.204_62,
    ("kg",   "g"):    lambda x: x * 1_000,
    ("g",    "kg"):   lambda x: x / 1_000,
    ("g",    "oz"):   lambda x: x / 28.349_5,
    ("oz",   "g"):    lambda x: x * 28.349_5,
    ("lb",   "oz"):   lambda x: x * 16,
    ("oz",   "lb"):   lambda x: x / 16,
    # 温度
    ("celsius",    "fahrenheit"): lambda x: x * 9 / 5 + 32,
    ("fahrenheit", "celsius"):    lambda x: (x - 32) * 5 / 9,
}

# ── 单位显示名（标准键 → 中文名）────────────────────────────────
_DISPLAY: dict[str, str] = {
    "km": "千米", "mile": "英里", "m": "米",
    "ft": "英尺", "cm":   "厘米", "inch": "英寸",
    "kg": "千克", "lb":   "磅",   "g":   "克",   "oz": "盎司",
    "celsius": "摄氏度", "fahrenheit": "华氏度",
}


def _normalize(unit: str) -> str:
    """将中英文单位别名统一为标准键（不区分大小写）。"""
    u = unit.strip()
    return _ALIASES.get(u, _ALIASES.get(u.lower(), u.lower()))


def execute(value: float, from_unit: str, to_unit: str) -> str:
    """
    执行单位换算。

    Args:
        value:     待换算数值。
        from_unit: 原单位（支持中英文，见 SKILL.md）。
        to_unit:   目标单位（支持中英文，见 SKILL.md）。

    Returns:
        换算结果字符串；单位不支持时返回错误说明及可用换算列表。
    """
    src = _normalize(from_unit)
    dst = _normalize(to_unit)

    # 相同单位无需换算
    if src == dst:
        src_name = _DISPLAY.get(src, src)
        return f"{value} {src_name} = {float(value):.4g} {src_name}（无需换算）"

    fn = _CONVERSIONS.get((src, dst))
    if fn is None:
        pairs = "、".join(f"{_DISPLAY.get(a,a)}↔{_DISPLAY.get(b,b)}" for a, b in _CONVERSIONS)
        return (
            f"不支持从「{from_unit}」到「{to_unit}」的换算。\n"
            f"支持的换算：{pairs}"
        )

    result = fn(float(value))
    src_name = _DISPLAY.get(src, src)
    dst_name = _DISPLAY.get(dst, dst)

    # 温度保留 2 位小数，其余保留 4 位有效数字
    if src in ("celsius", "fahrenheit"):
        return f"{value} {src_name} = {result:.2f} {dst_name}"
    return f"{value} {src_name} = {result:.4g} {dst_name}"
