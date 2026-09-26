"""Workspace root and writable-path resolution."""
from __future__ import annotations

import os
from pathlib import Path


def _compute_workspace_root() -> Path:
    override = os.environ.get("STRATUNE_WORKSPACE", "").strip()
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return Path(os.environ.get("STRATUNE_ROOT") or Path(__file__).resolve().parents[3]).resolve(strict=False)


WORKSPACE_ROOT = _compute_workspace_root()


def resolve_writable_path(path: str | os.PathLike[str]) -> Path:
    """Relative paths are interpreted relative to the workspace root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    return candidate.resolve(strict=False)
