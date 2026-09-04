from __future__ import annotations
import os, logging, threading
from datetime import datetime, timezone, timedelta

log = logging.getLogger("app.key_manager")

_ROTATION_THRESHOLD   = 20
_EXHAUSTION_THRESHOLD = 40
_COOLDOWN_HOURS       = 24
_NUM_KEYS             = 20
_lock = threading.Lock()
_ALL_KEYS = []


def _load_keys():
    keys = []
    for i in range(_NUM_KEYS):
        k = os.getenv(f"GOOGLE_API_KEY_{i:02d}", "").strip()
        if k and k != "your-gemini-api-key-here":
            keys.append(k)
    if not keys:
        fb = os.getenv("GOOGLE_API_KEY", "").strip()
        if fb and fb != "your-gemini-api-key-here":
            keys.append(fb)
    return keys


def _keys():
    global _ALL_KEYS
    if not _ALL_KEYS:
        _ALL_KEYS = _load_keys()
        log.info(f"[KeyManager] Loaded {len(_ALL_KEYS)} Gemini API keys")
    return _ALL_KEYS


def _get_state():
    try:
        from app.mongodb import get_key_state
        s = get_key_state()
        if s:
            return s
    except Exception:
        pass
    return {"current_key_index": 0, "quota_fail_count": 0,
            "exhausted_at": None, "key_fail_counts": {}}


def _save_state(patch):
    try:
        from app.mongodb import update_key_state
        update_key_state(patch)
    except Exception:
        pass


def get_active_key():
    """Returns (api_key, key_index). Returns (None, -1) when all keys exhausted."""
    with _lock:
        keys = _keys()
        if not keys:
            log.error("[KeyManager] No API keys configured!")
            return None, -1
        state = _get_state()
        fail_count = state.get("quota_fail_count", 0)
        ea_str = state.get("exhausted_at")
        idx = state.get("current_key_index", 0) % len(keys)
        if ea_str:
            try:
                ea = datetime.fromisoformat(ea_str)
                if ea.tzinfo is None:
                    ea = ea.replace(tzinfo=timezone.utc)
                elapsed = datetime.now(timezone.utc) - ea
                if elapsed >= timedelta(hours=_COOLDOWN_HOURS):
                    log.info("[KeyManager] 24hr cooldown expired. Resetting.")
                    _save_state({"current_key_index": 0, "quota_fail_count": 0,
                                 "exhausted_at": None, "key_fail_counts": {}})
                    return keys[0], 0
                else:
                    return None, -1
            except Exception:
                pass
        if fail_count >= _EXHAUSTION_THRESHOLD:
            _save_state({"exhausted_at": datetime.now(timezone.utc).isoformat()})
            log.error(f"[KeyManager] All keys exhausted (count={fail_count}). Cooldown started.")
            return None, -1
        return keys[idx], idx


def report_quota_failure(key_index):
    """Called on quota error. Increments counter, rotates to next key."""
    with _lock:
        keys = _keys()
        state = _get_state()
        fail_count = state.get("quota_fail_count", 0) + 1
        kf = dict(state.get("key_fail_counts", {}))
        ks = str(key_index)
        kf[ks] = kf.get(ks, 0) + 1
        if fail_count <= _ROTATION_THRESHOLD:
            new_idx = (key_index + 1) % len(keys)
        elif fail_count == _ROTATION_THRESHOLD + 1:
            new_idx = 0
            log.warning(f"[KeyManager] {_ROTATION_THRESHOLD} failures. Wrapping to key 0.")
        else:
            new_idx = (key_index + 1) % len(keys)
        exhausted = fail_count >= _EXHAUSTION_THRESHOLD
        patch = {"current_key_index": new_idx, "quota_fail_count": fail_count,
                 "key_fail_counts": kf}
        if exhausted:
            patch["exhausted_at"] = datetime.now(timezone.utc).isoformat()
        _save_state(patch)
        log.warning(f"[KeyManager] Quota #{fail_count} key[{key_index}]->key[{new_idx}] exhausted={exhausted}")
        return {"exhausted": exhausted,
                "hours_remaining": float(_COOLDOWN_HOURS) if exhausted else 0.0,
                "new_key_index": new_idx,
                "fail_count": fail_count}


def report_success(key_index):
    pass  # Quota is per-day; no reset on success


def is_exhausted():
    """Returns (is_exhausted: bool, hours_remaining: float)."""
    state = _get_state()
    fc = state.get("quota_fail_count", 0)
    ea_str = state.get("exhausted_at")
    if fc < _EXHAUSTION_THRESHOLD or not ea_str:
        return False, 0.0
    try:
        ea = datetime.fromisoformat(ea_str)
        if ea.tzinfo is None:
            ea = ea.replace(tzinfo=timezone.utc)
        elapsed = datetime.now(timezone.utc) - ea
        if elapsed >= timedelta(hours=_COOLDOWN_HOURS):
            return False, 0.0
        remaining = (timedelta(hours=_COOLDOWN_HOURS) - elapsed).total_seconds() / 3600
        return True, round(remaining, 1)
    except Exception:
        return False, 0.0


def get_status():
    """Returns current key rotation status for admin/health endpoints."""
    keys = _keys()
    state = _get_state()
    exhausted, hrs = is_exhausted()
    return {
        "total_keys": len(keys),
        "current_key_index": state.get("current_key_index", 0),
        "quota_fail_count": state.get("quota_fail_count", 0),
        "exhausted": exhausted,
        "hours_remaining_in_cooldown": hrs,
        "key_fail_counts": state.get("key_fail_counts", {}),
    }
