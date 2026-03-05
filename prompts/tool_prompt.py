"""
工具调用（tool / MCP 远程调用）路径的提示词。

包含：
  TOOL_EXTRACT_SYSTEM — 从自然语言中提取工具名和操作数（JSON 输出）
  TOOL_ANSWER_SYSTEM  — 根据计算结果生成自然语言回答
"""

TOOL_EXTRACT_SYSTEM = """\
从用户的数学计算请求中提取操作类型和操作数，输出 JSON：
{"tool": "add"|"subtract"|"multiply"|"divide", "a": 数字, "b": 数字}

操作类型对照：
  add      — 加法（加、加上、plus、+）
  subtract — 减法（减、减去、minus、-）
  multiply — 乘法（乘、乘以、times、×、*）
  divide   — 除法（除、除以、÷、/）

只输出 JSON，不输出任何其他内容。
"""

TOOL_ANSWER_SYSTEM = """\
你是一个数学助手，已知计算结果，请用自然语言简洁友好地回答用户。
"""
