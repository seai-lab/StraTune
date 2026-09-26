"""Exact-match JSONL cache for model responses (thread-safe, append-only)."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

from .paths import resolve_writable_path


class CacheRequest:
    """Canonical description of one model request; identical requests share one cache key."""

    def __init__(self, *, system_bytes: bytes, user_bytes: bytes, model: str, decoding_config: dict,
                 artifact_hash: str, deployment_hash: str, seed: int):
        self.system_bytes, self.user_bytes, self.model = system_bytes, user_bytes, model
        self.decoding_config = dict(decoding_config)
        self.artifact_hash, self.deployment_hash, self.seed = artifact_hash, deployment_hash, int(seed)

    @property
    def key(self) -> str:
        h = hashlib.sha256()
        for part in (self.system_bytes, self.user_bytes, self.model.encode(),
                     json.dumps(self.decoding_config, sort_keys=True).encode(),
                     self.artifact_hash.encode(), self.deployment_hash.encode(), str(self.seed).encode()):
            h.update(hashlib.sha256(part).digest())
        return h.hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    key: str
    response: Any


class ExactJSONLCache:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = resolve_writable_path(path)
        self._lock = threading.RLock()
        self._entries: dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._entries[rec["key"]] = rec["response"]

    def lookup(self, request: CacheRequest) -> CacheEntry | None:
        with self._lock:
            if request.key in self._entries:
                return CacheEntry(request.key, self._entries[request.key])
        return None

    def put(self, request: CacheRequest, response: Any) -> CacheEntry:
        with self._lock:
            self._entries[request.key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": request.key, "response": response}, ensure_ascii=False) + "\n")
        return CacheEntry(request.key, response)

    def __len__(self) -> int:
        return len(self._entries)
