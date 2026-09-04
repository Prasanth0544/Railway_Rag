"""
app/mongodb.py
Centralised MongoDB Atlas connection + helpers.
Both local and Render deployments write to the same Atlas cluster.

Database : railway_rag
Collections:
  query_logs  - one doc per user query
  feedback    - one doc per thumbs rating
"""
from __future__ import annotations
import logging, os, threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure

log = logging.getLogger("app.mongodb")

DB_NAME       = "Railway_Rag"
COL_QUERIES   = "query_logs"
COL_FEEDBACK  = "feedback"
COL_KEY_STATE = "api_key_state"

_client = None
_db     = None
_status = "not_configured"


def _connect():
    global _client, _db, _status
    # Read fresh from env each time (must be inside function so dotenv is loaded first)
    mongo_uri = os.getenv("MONGO_URI", "")
    if not mongo_uri:
        _status = "not_configured"
        log.warning("[MongoDB] MONGO_URI not set - disabled.")
        return
    _status = "connecting"
    log.info("[MongoDB] Connecting to Atlas...")
    try:
        _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
        _db = _client[DB_NAME]
        _client.admin.command("ping")
        _status = "online"
        log.info("[MongoDB] Connected to Atlas, db=%s", DB_NAME)
        _ensure_indexes()
    except (ConnectionFailure, ServerSelectionTimeoutError, OperationFailure) as e:
        _status = "error"
        log.error("[MongoDB] Connection failed: %s", e)
        _client = None
        _db = None


def _ensure_indexes():
    try:
        _db[COL_QUERIES].create_index([("ts", DESCENDING)])
        _db[COL_QUERIES].create_index([("intent", ASCENDING)])
        _db[COL_QUERIES].create_index([("question", ASCENDING)])
        _db[COL_FEEDBACK].create_index([("ts", DESCENDING)])
        _db[COL_FEEDBACK].create_index([("rating", ASCENDING)])
        _db[COL_KEY_STATE].create_index([("_id", ASCENDING)], unique=True)
        log.info("[MongoDB] Indexes verified.")
    except Exception as e:
        log.warning("[MongoDB] Index warning: %s", e)


def init_async():
    """Connect in a background thread - never blocks server startup."""
    threading.Thread(target=_connect, name="mongo-init", daemon=True).start()


def is_online():
    return _status == "online" and _client is not None


def get_status():
    return _status


def ping():
    if _client is None:
        return False
    try:
        _client.admin.command("ping")
        return True
    except Exception:
        return False


def insert_query_log(doc: Dict[str, Any]) -> bool:
    if not is_online():
        return False
    try:
        doc.setdefault("ts", datetime.now(timezone.utc))
        _db[COL_QUERIES].insert_one(doc)
        return True
    except Exception as e:
        log.error("[MongoDB] insert_query_log: %s", e)
        return False


def insert_feedback(doc: Dict[str, Any]) -> bool:
    if not is_online():
        return False
    try:
        doc.setdefault("ts", datetime.now(timezone.utc))
        _db[COL_FEEDBACK].insert_one(doc)
        return True
    except Exception as e:
        log.error("[MongoDB] insert_feedback: %s", e)
        return False


def update_feedback_comment(session_id: str, question: str, rating: str, comment: str) -> bool:
    """
    YouTube-style comment update: finds the most recent feedback record for
    this session + question + rating and patches its comment field.
    Does NOT insert a new document, so rating counts remain accurate (1 per action).
    Returns True if a record was found and updated, False otherwise.
    """
    if not is_online():
        return False
    try:
        from pymongo import DESCENDING
        result = _db[COL_FEEDBACK].find_one_and_update(
            {"session_id": session_id, "question": question, "rating": rating},
            {"$set": {"comment": comment, "commented_at": datetime.now(timezone.utc)}},
            sort=[("ts", DESCENDING)],
        )
        return result is not None
    except Exception as e:
        log.error("[MongoDB] update_feedback_comment: %s", e)
        return False


def get_feedback_summary() -> Optional[Dict[str, Any]]:
    if not is_online():
        return None
    try:
        col = _db[COL_FEEDBACK]
        total = col.count_documents({})
        if not total:
            return {"total": 0, "thumbs_up": 0, "thumbs_down": 0,
                    "positive_rate_pct": 0, "top_liked": [], "top_disliked": []}
        up   = col.count_documents({"rating": "up"})
        down = col.count_documents({"rating": "down"})
        pct  = round(up / total * 100)
        top_liked = [
            {"question": d["question"], "count": 1}
            for d in col.find(
                {"rating": "up", "question": {"$exists": True, "$ne": ""}},
                {"question": 1, "_id": 0}
            ).sort("ts", DESCENDING).limit(3)
        ]
        top_disliked = [
            {"question": d["question"], "count": 1}
            for d in col.find(
                {"rating": "down", "question": {"$exists": True, "$ne": ""}},
                {"question": 1, "_id": 0}
            ).sort("ts", DESCENDING).limit(3)
        ]
        return {
            "total": total, "thumbs_up": up, "thumbs_down": down,
            "positive_rate_pct": pct,
            "top_liked": top_liked, "top_disliked": top_disliked,
        }
    except Exception as e:
        log.error("[MongoDB] get_feedback_summary: %s", e)
        return None


def get_query_analytics() -> Optional[Dict[str, Any]]:
    if not is_online():
        return None
    try:
        col = _db[COL_QUERIES]
        total = col.count_documents({})
        if not total:
            return None
        intent_dist = {
            (d["_id"] or "unknown"): d["count"]
            for d in col.aggregate([
                {"$group": {"_id": "$intent", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ])
        }
        avg_docs = list(col.aggregate([
            {"$match": {"response_time_ms": {"$exists": True, "$gt": 0}}},
            {"$group": {"_id": None, "avg": {"$avg": "$response_time_ms"}}},
        ]))
        avg_ms = round(avg_docs[0]["avg"]) if avg_docs else 0
        top_qs = [
            {"question": d["_id"], "count": d["count"]}
            for d in col.aggregate([
                {"$match": {"question": {"$exists": True, "$ne": ""}}},
                {"$group": {"_id": "$question", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 8},
            ])
        ]
        errors  = col.count_documents({"error": True})
        success = total - errors
        live    = col.count_documents({"used_live_api": True})
        live_ok = col.count_documents({"used_live_api": True, "error": {"$ne": True}})

        # Error type breakdown (why queries failed)
        error_type_breakdown = {
            (d["_id"] or "server_error"): d["count"]
            for d in col.aggregate([
                {"$match": {"error": True, "error_type": {"$exists": True}}},
                {"$group": {"_id": "$error_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ])
        }

        # Recent failed queries (last 5) with reason
        recent_failures = [
            {
                "question": d.get("question", "")[:100],
                "error_type": d.get("error_type", "server_error"),
                "error_reason": d.get("error_reason", "")[:120],
                "ts": d.get("ts", ""),
            }
            for d in col.find(
                {"error": True},
                {"question": 1, "error_type": 1, "error_reason": 1, "ts": 1, "_id": 0}
            ).sort("ts", DESCENDING).limit(5)
        ]

        return {
            "total_queries": total,
            "success_queries": success,
            "failed_queries": errors,
            "success_rate_pct": round(success / total * 100, 1) if total else 100,
            "avg_response_time_ms": avg_ms,
            "error_rate_pct": round(errors / total * 100, 1) if total else 0,
            "error_type_breakdown": error_type_breakdown,
            "recent_failures": recent_failures,
            "intent_distribution": intent_dist,
            "top_questions": top_qs,
            "live_api_used": live,
            "live_api_success_rate_pct": round(live_ok / live * 100) if live else 100,
            "hallucination_flags": col.count_documents({"hallucination_flag": True}),
            "source": "mongodb",
        }
    except Exception as e:
        log.error("[MongoDB] get_query_analytics: %s", e)
        return None


# ── API Key State (for key rotation manager) ─────────────────────────────────
_KEY_STATE_ID = "gemini_rotation"

def get_key_state() -> Optional[Dict[str, Any]]:
    """Read the current key rotation state from MongoDB."""
    if not is_online():
        return None
    try:
        doc = _db[COL_KEY_STATE].find_one({"_id": _KEY_STATE_ID})
        if doc:
            doc.pop("_id", None)
            return doc
        return None
    except Exception as e:
        log.error("[MongoDB] get_key_state: %s", e)
        return None


def update_key_state(patch: Dict[str, Any]) -> bool:
    """Upsert the key rotation state document with the given fields."""
    if not is_online():
        return False
    try:
        _db[COL_KEY_STATE].update_one(
            {"_id": _KEY_STATE_ID},
            {"$set": patch},
            upsert=True,
        )
        return True
    except Exception as e:
        log.error("[MongoDB] update_key_state: %s", e)
        return False
