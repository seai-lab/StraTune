"""Shared helpers: canonical JSON, hashing, atomic writes, environment setup."""
import hashlib
import json
import os
import tempfile


WORKDIR = os.environ.get("STRATUNE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_ROOT = os.path.join(WORKDIR, "runtime_data")


def setup_env():
    """Place TMPDIR and temporary files under data_root/tmp.

    Credentials are not configured here; the launching script sets them.
    """
    tmp = os.path.join(DATA_ROOT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    os.environ["TMPDIR"] = tmp
    tempfile.tempdir = tmp


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    """Hash of an ID list, using the same JSON encoding as the split manifests."""
    return sha256_str(json.dumps(list(ids), separators=(",", ":")))


def atomic_write_json(path: str, obj, indent=1):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def read_json(path: str):
    with open(path) as f:
        return json.load(f)
