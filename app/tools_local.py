from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import re
from typing import Any, Dict

def calc(expression: str) -> Dict[str, Any]:
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expression):
        return {"ok": False, "error": "Invalid characters in expression."}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        if isinstance(result, (int, float)) and math.isfinite(result):
            return {"ok": True, "result": result}
        return {"ok": False, "error": "Non-finite result."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def get_time(tz: str = "Asia/Dubai") -> Dict[str, Any]:
    try:
        now = datetime.now(ZoneInfo(tz))
        return {"ok": True, "timezone": tz, "iso": now.isoformat()}
    except Exception as e:
        return {"ok": False, "error": f"Invalid timezone: {tz}. {e}"}

def summarize(text: str, max_bullets: int = 5) -> Dict[str, Any]:
    bullets = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = bullets[:max_bullets]
    return {"ok": True, "bullets": bullets}
