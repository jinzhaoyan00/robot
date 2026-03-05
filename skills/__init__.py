"""
Skills 注册与调度模块（目录加载版）。

每个技能以独立子目录形式存放，遵循 awesome-agent-skills 标准结构：

    skills/
    └── <skill-name>/
        ├── SKILL.md          ← 技能元数据（YAML frontmatter）+ 文档
        └── scripts/
            └── execute.py    ← 执行逻辑，须导出 execute(**kwargs) -> str

SKILL.md frontmatter 必填字段：
  name         — 技能标识符（即调度键）
  description  — 技能功能描述（供意图分类提示词使用）
  params_hint  — JSON 参数示例（供参数提取提示词使用）

对外接口：
  REGISTRY            — {name: SkillEntry} 字典
  get_intent_hint()   — 返回技能列表描述，供意图分类提示词使用
  get_extract_hint()  — 返回参数提取提示词片段
  execute_skill(name, params) — 按名称调度执行技能
"""

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_SKILLS_DIR = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════

@dataclass
class SkillEntry:
    """单个技能的元数据 + 执行函数。"""
    name:        str
    description: str
    params_hint: str
    execute_fn:  Callable


# ══════════════════════════════════════════════════════════════════
# SKILL.md frontmatter 解析
# ══════════════════════════════════════════════════════════════════

def _parse_frontmatter(md_path: Path) -> dict[str, str]:
    """
    从 SKILL.md 中解析 YAML frontmatter（--- ... --- 之间的内容）。

    只处理简单的 key: value 形式，值会去除首尾引号和空白。
    Returns:
        解析得到的字典；解析失败时返回空字典。
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    if not text.startswith("---"):
        return {}

    end = text.find("\n---", 3)
    if end == -1:
        return {}

    fm_text = text[4:end]          # 跳过首行 "---\n"
    result: dict[str, str] = {}
    for line in fm_text.splitlines():
        m = re.match(r"^(\w+)\s*:\s*(.+)$", line.strip())
        if m:
            key = m.group(1)
            val = m.group(2).strip().strip("'\"")
            result[key] = val
    return result


# ══════════════════════════════════════════════════════════════════
# 技能目录自动发现与加载
# ══════════════════════════════════════════════════════════════════

def _load_execute_fn(skill_dir: Path, skill_name: str) -> Callable | None:
    """
    从 <skill_dir>/scripts/execute.py 动态加载并返回 execute 函数。
    加载失败时打印警告并返回 None。
    """
    execute_py = skill_dir / "scripts" / "execute.py"
    if not execute_py.exists():
        return None

    module_key = f"skills._exec.{skill_name}"
    if module_key in sys.modules:
        return getattr(sys.modules[module_key], "execute", None)

    spec = importlib.util.spec_from_file_location(module_key, execute_py)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    try:
        spec.loader.exec_module(module)          # type: ignore[attr-defined]
    except Exception as exc:
        print(f"[Skills] 加载技能 '{skill_name}' 失败: {exc}")
        del sys.modules[module_key]
        return None

    return getattr(module, "execute", None)


def _discover_skills() -> dict[str, SkillEntry]:
    """
    扫描 skills/ 目录，加载所有合法技能并返回注册表字典。

    合法技能目录须同时包含：
      - SKILL.md（含 name / description / params_hint frontmatter）
      - scripts/execute.py（含 execute() 函数）
    """
    registry: dict[str, SkillEntry] = {}

    for skill_dir in sorted(_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith(("_", ".")):
            continue

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        meta = _parse_frontmatter(skill_md)
        name        = meta.get("name",        skill_dir.name)
        description = meta.get("description", "")
        params_hint = meta.get("params_hint", "")

        if not description:
            print(f"[Skills] 跳过 '{skill_dir.name}'：SKILL.md 缺少 description 字段")
            continue

        execute_fn = _load_execute_fn(skill_dir, name)
        if execute_fn is None:
            print(f"[Skills] 跳过 '{skill_dir.name}'：scripts/execute.py 缺失或加载失败")
            continue

        registry[name] = SkillEntry(
            name=name,
            description=description,
            params_hint=params_hint,
            execute_fn=execute_fn,
        )

    return registry


# 模块加载时自动发现所有技能
REGISTRY: dict[str, SkillEntry] = _discover_skills()


# ══════════════════════════════════════════════════════════════════
# 对外接口
# ══════════════════════════════════════════════════════════════════

def get_intent_hint() -> str:
    """返回供意图分类提示词使用的技能列表描述。"""
    lines = [f"    {e.name}: {e.description}" for e in REGISTRY.values()]
    return "\n".join(lines)


def get_extract_hint() -> str:
    """返回供参数提取提示词使用的 JSON 示例列表。"""
    lines = [f"  {e.name}: {e.params_hint}" for e in REGISTRY.values() if e.params_hint]
    return "\n".join(lines)


def execute_skill(name: str, params: dict) -> str:
    """
    按技能名称执行技能。

    Args:
        name:   技能名称（对应 REGISTRY 中的键）。
        params: 传给技能 execute() 的关键字参数字典。

    Returns:
        技能执行结果字符串；技能不存在或执行出错时返回错误说明。
    """
    entry = REGISTRY.get(name)
    if entry is None:
        available = "、".join(REGISTRY.keys()) or "（暂无已加载的技能）"
        return f"未知技能「{name}」，可用技能：{available}"
    try:
        return entry.execute_fn(**params)
    except TypeError as exc:
        return f"技能参数错误：{exc}"
    except Exception as exc:
        return f"技能执行失败：{exc}"
