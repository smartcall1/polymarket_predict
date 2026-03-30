"""
Decision Logger — AI 의사결정 과정을 JSONL로 기록.

각 마켓 분석의 전체 reasoning, 확률, edge, 필터 결과를 저장.
용량: ~10-20KB/건, 하루 ~0.5-1MB (부담 없음).

파일: data/decisions_YYYYMMDD.jsonl (일별 분리, 자동 정리)
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _get_log_path() -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(DATA_DIR, f"decisions_{today}.jsonl")


def log_decision(
    market_id: str,
    market_title: str,
    action: str,
    side: str = "",
    yes_price: float = 0,
    no_price: float = 0,
    confidence: float = 0,
    edge: float = 0,
    reasoning: str = "",
    filter_reason: str = "",
    ai_cost: float = 0,
    volume: int = 0,
    extra: Optional[Dict[str, Any]] = None,
):
    """Append a decision record to today's JSONL log."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "market_id": market_id[:20],
        "title": market_title[:100],
        "action": action,
        "side": side,
        "yes_price": round(yes_price, 4),
        "no_price": round(no_price, 4),
        "confidence": round(confidence, 4),
        "edge": round(edge, 4),
        "reasoning": reasoning[:500],
        "filter_reason": filter_reason,
        "ai_cost": round(ai_cost, 6),
        "volume": volume,
    }
    if extra:
        record["extra"] = extra

    try:
        with open(_get_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_today_decisions(limit: int = 50) -> list:
    """Read today's decision log."""
    path = _get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-limit:]]
    except Exception:
        return []


def get_today_stats() -> Dict:
    """Quick stats from today's decisions."""
    decisions = get_today_decisions(limit=9999)
    if not decisions:
        return {"total": 0, "buys": 0, "skips": 0, "cost": 0}

    buys = [d for d in decisions if d.get("action", "").upper() == "BUY"]
    skips = [d for d in decisions if d.get("action", "").upper() != "BUY"]
    total_cost = sum(d.get("ai_cost", 0) for d in decisions)

    return {
        "total": len(decisions),
        "buys": len(buys),
        "skips": len(skips),
        "cost": round(total_cost, 4),
    }
