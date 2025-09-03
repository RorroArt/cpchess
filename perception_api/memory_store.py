# memory_store.py
# JSON-backed memory with safe atomic writes. Designed to be instantiated
# with a path (e.g., .../live_outputs/board_memory.json).

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
import os

BoardPiece = Optional[Tuple[str, str]]  # ("K","White") or None for empty


def _now_iso():
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def _safe_write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # atomic on POSIX


@dataclass
class _State:
    version: int = 2
    last_committed_board: Dict[str, BoardPiece] = field(default_factory=dict)
    last_confidence: Dict[str, float] = field(default_factory=dict)
    last_cosine: Dict[str, float] = field(default_factory=dict)
    pending_board: Optional[Dict[str, BoardPiece]] = None
    pending_confidence: Optional[Dict[str, float]] = None
    pending_cosine: Optional[Dict[str, float]] = None
    start_bias_locked: bool = False
    start_bias_since: Optional[str] = None
    last_committed_time: Optional[str] = None
    last_pending_time: Optional[str] = None


class Memory:
    """
    JSON-backed memory. Thread/process-friendly via atomic replace on save.
    Reads are tolerant to partial failures. Minimal schema migration support.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = _State()
        if self.path.exists():
            self._load()

    # ---------- persistence ----------
    def _load(self):
        try:
            raw = json.loads(self.path.read_text())
        except Exception:
            # If the file was partially written, try the .tmp fallback
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            if tmp.exists():
                try:
                    raw = json.loads(tmp.read_text())
                except Exception:
                    raw = {}
            else:
                raw = {}

        # Merge into state, tolerating missing fields
        s = asdict(self.state)
        s.update(raw if isinstance(raw, dict) else {})
        # Backward-compatible additions
        s.setdefault("last_cosine", {})
        s.setdefault("pending_cosine", None)
        self.state = _State(**s)

    def _save(self):
        data = asdict(self.state)
        _safe_write_json(self.path, data)

    # ---------- helpers ----------
    @property
    def start_bias_locked(self) -> bool:
        return bool(self.state.start_bias_locked)

    def lock_start_bias(self):
        self.state.start_bias_locked = True
        self.state.start_bias_since = _now_iso()
        self._save()

    def set_pending(
        self,
        board_map: Dict[str, BoardPiece],
        conf_map: Dict[str, float],
        cos_map: Dict[str, float],
    ):
        self.state.pending_board = {k: v for k, v in board_map.items()}
        self.state.pending_confidence = dict(conf_map)
        self.state.pending_cosine = dict(cos_map)
        self.state.last_pending_time = _now_iso()
        self._save()

    def commit_pending(self, updated_board=None, updated_conf=None, updated_cos=None):
        if self.state.pending_board is None:
            return
        board = updated_board if updated_board is not None else self.state.pending_board
        confs = updated_conf if updated_conf is not None else (self.state.pending_confidence or {})
        cosns = updated_cos if updated_cos is not None else (self.state.pending_cosine or {})

        self.state.last_committed_board = {k: v for k, v in board.items()}
        self.state.last_confidence = dict(confs)
        self.state.last_cosine = dict(cosns)
        self.state.last_committed_time = _now_iso()

        # clear pending
        self.state.pending_board = None
        self.state.pending_confidence = None
        self.state.pending_cosine = None
        self._save()

    def last_committed(self):
        b = self.state.last_committed_board or {}
        c = self.state.last_confidence or {}
        s = self.state.last_cosine or {}
        return dict(b), dict(c), dict(s)

    def pending(self):
        b = self.state.pending_board
        c = self.state.pending_confidence
        s = self.state.pending_cosine
        if b is None:
            return None, None, None
        return dict(b), dict(c or {}), dict(s or {})

    # Utility to reset file
    def reset(self):
        self.state = _State()
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._save()