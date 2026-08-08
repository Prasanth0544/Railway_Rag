"""
analytics.py — Query Analytics Logger

Logs every query with intent, retrieval stats, response time, and errors
to a JSONL file for offline analysis.
"""

import json
import os
import time
import threading
from typing import Any

from app.logger import get_logger
logger = get_logger("app.analytics")

# Log file path — in the project root
ANALYTICS_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "query_log.jsonl")

_write_lock = threading.Lock()


def log_query(
    question: str,
    intent: str,
    confidence: float,
    train_no: str | None,
    num_docs: int,
    avg_score: float,
    response_time_ms: float,
    source: str = "unknown",
    live_api_used: bool = False,
    live_api_success: bool = False,
    error: str | None = None,
    validation_warnings: list[str] | None = None,
    context_strategy: str = "default",     # smart_format_docs strategy chosen
) -> None:
    """
    Append a query analytics entry to the JSONL log file.
    Thread-safe via write lock.
    """
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": question[:200],  # truncate long queries
        "intent": intent,
        "confidence": confidence,
        "train_no": train_no,
        "num_docs": num_docs,
        "avg_relevance_score": avg_score,
        "response_time_ms": response_time_ms,
        "source": source,
        "live_api_used": live_api_used,
        "live_api_success": live_api_success,
        "error": error,
        "validation_warnings": validation_warnings or [],
        "context_strategy": context_strategy,
    }

    try:
        with _write_lock:
            with open(ANALYTICS_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to write log: {e}")


def get_stats() -> dict[str, Any]:
    """
    Read the analytics log and compute summary statistics.
    Returns totals, averages, top queries, and error rates.
    """
    if not os.path.exists(ANALYTICS_LOG_FILE):
        return {"total_queries": 0, "message": "No queries logged yet."}

    entries = []
    try:
        with open(ANALYTICS_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as e:
        return {"error": f"Failed to read log: {e}"}

    if not entries:
        return {"total_queries": 0, "message": "No queries logged yet."}

    total = len(entries)
    avg_response_ms = sum(e.get("response_time_ms", 0) for e in entries) / total
    error_count = sum(1 for e in entries if e.get("error"))
    live_used = sum(1 for e in entries if e.get("live_api_used"))
    live_success = sum(1 for e in entries if e.get("live_api_success"))
    hallucination_flags = sum(1 for e in entries if e.get("validation_warnings"))

    # Intent distribution
    intent_counts: dict[str, int] = {}
    for e in entries:
        intent = e.get("intent", "UNKNOWN")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    # Top 10 most asked questions (normalized)
    question_counts: dict[str, int] = {}
    for e in entries:
        q = e.get("question", "").lower().strip()[:100]
        question_counts[q] = question_counts.get(q, 0) + 1
    top_questions = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Context strategy distribution
    strategy_counts: dict[str, int] = {}
    for e in entries:
        s = e.get("context_strategy", "default")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    return {
        "total_queries": total,
        "avg_response_time_ms": round(avg_response_ms, 1),
        "error_count": error_count,
        "error_rate_pct": round(error_count / total * 100, 1) if total > 0 else 0,
        "live_api_used": live_used,
        "live_api_success_rate_pct": round(live_success / live_used * 100, 1) if live_used > 0 else 0,
        "hallucination_flags": hallucination_flags,
        "intent_distribution": intent_counts,
        "context_strategy_distribution": strategy_counts,
        "top_questions": [{"question": q, "count": c} for q, c in top_questions],
    }
