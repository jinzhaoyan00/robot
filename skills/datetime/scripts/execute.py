"""
日期时间查询技能 — 执行脚本。

入口函数：execute(query_type="full") -> str
"""

from datetime import datetime

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def execute(query_type: str = "full") -> str:
    """
    返回当前日期/时间信息。

    Args:
        query_type:
            full    — 完整日期时间 + 星期（默认）
            date    — 仅日期（年月日）
            time    — 仅时间（时分秒）
            weekday — 仅星期几

    Returns:
        格式化的中文日期/时间字符串。
    """
    now = datetime.now()
    weekday = _WEEKDAYS[now.weekday()]

    if query_type == "date":
        return now.strftime("%Y年%m月%d日")
    if query_type == "time":
        return now.strftime("%H:%M:%S")
    if query_type == "weekday":
        return weekday
    return f"{now.strftime('%Y年%m月%d日')} {now.strftime('%H:%M:%S')} {weekday}"
