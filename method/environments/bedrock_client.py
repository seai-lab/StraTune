"""Bedrock LLM access shared by the optimizer LLM and the target LLM.

Self-contained module (no imports from the runtime package). Provides:

  * ``fail_fast_identity()`` -- verifies that AWS credentials resolve (STS
    get-caller-identity) before any model call; optionally checks that the
    caller ARN contains STRATUNE_EXPECTED_ROLE.
  * ``BedrockConverseClient`` -- thin, deterministic wrapper over the Bedrock
    Converse API with temperature-0 defaults, seed-awareness (seed is part of
    the cache key; Anthropic models on Bedrock do not accept a seed
    parameter), PNG image support, long read timeouts, adaptive retries with
    exponential backoff, and an abort after 5 *consecutive*
    credential-shaped failures.
  * ``JsonlCache`` -- thread-safe, append-only exact-match JSONL cache keyed
    by sha256 of the canonical request; fail-closed on key collisions.

Default model ids (us-west-2):
  target LLM   : global.anthropic.claude-haiku-4-5-20251001-v1:0
  optimizer LLM: global.anthropic.claude-sonnet-4-6
"""

from __future__ import annotations

import os


import hashlib
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger("stratune.llm")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGION = "us-west-2"

TARGET_MODEL = os.environ.get("STRATUNE_TARGET_MODEL",
                              "global.anthropic.claude-haiku-4-5-20251001-v1:0")
OPTIMIZER_MODEL = os.environ.get("STRATUNE_OPTIMIZER_MODEL", "global.anthropic.claude-sonnet-4-6")
# Model families that reject sampling parameters (temperature/top_p -> 400).
_NO_SAMPLING = ("opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable")


EXPECTED_ROLE_FRAGMENT = os.environ.get("STRATUNE_EXPECTED_ROLE", "")

REMEDIATION_HINT = (
    "Credential remediation: configure AWS credentials with Amazon Bedrock access "
    "(AWS_PROFILE / AWS_ACCESS_KEY_ID or AWS_BEARER_TOKEN_BEDROCK) and set AWS_REGION."
)

# Error codes that indicate broken/expired credentials rather than a
# transient service problem.
CREDENTIAL_ERROR_CODES = {
    "ExpiredToken",
    "ExpiredTokenException",
    "InvalidClientTokenId",
    "UnrecognizedClientException",
    "InvalidSignatureException",
    "AccessDeniedException",  # often what an expired token degrades to
}

# Transient errors worth retrying with backoff (on top of botocore adaptive
# retries, which already handle most throttling).
TRANSIENT_ERROR_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
    "ModelNotReadyException",
    "ServiceQuotaExceededException",
}

MAX_CONSECUTIVE_CREDENTIAL_FAILURES = 5


class CredentialFailureAbort(RuntimeError):
    """Raised after too many consecutive credential-shaped failures."""


class CacheCollisionError(RuntimeError):
    """Raised when a cache key maps to a different canonical request.

    Fail-closed: we never return a cached response whose stored request does
    not byte-match the current canonical request.
    """


# ---------------------------------------------------------------------------
# Identity check
# ---------------------------------------------------------------------------

def fail_fast_identity(region: str = REGION) -> str:
    """Verify that STS resolves a caller identity (and, if STRATUNE_EXPECTED_ROLE is set, that the ARN contains it).

    Returns the caller ARN on success; raises RuntimeError with a
    remediation hint otherwise.
    """
    try:
        sts = boto3.client("sts", region_name=region)
        ident = sts.get_caller_identity()
    except Exception as exc:  # noqa: BLE001 - anything here means no creds
        raise RuntimeError(
            f"STS get-caller-identity failed: {exc}\n{REMEDIATION_HINT}"
        ) from exc
    arn = ident.get("Arn", "")
    if EXPECTED_ROLE_FRAGMENT and EXPECTED_ROLE_FRAGMENT not in arn:
        raise RuntimeError(
            f"Unexpected AWS identity {arn!r}; expected role containing "
            f"{EXPECTED_ROLE_FRAGMENT!r}.\n{REMEDIATION_HINT}"
        )
    logger.debug("AWS identity OK: %s", arn)
    return arn


# ---------------------------------------------------------------------------
# Content-block helpers (Converse API shapes)
# ---------------------------------------------------------------------------

def text_block(text: str) -> dict:
    return {"text": text}


def image_block_png(png_bytes: bytes) -> dict:
    """PNG bytes -> Converse image content block."""
    if not isinstance(png_bytes, (bytes, bytearray)):
        raise TypeError("image_block_png expects raw PNG bytes")
    return {"image": {"format": "png", "source": {"bytes": bytes(png_bytes)}}}


def user_message(*blocks: Any) -> dict:
    """Build a user message from strings and/or content-block dicts."""
    content = [text_block(b) if isinstance(b, str) else b for b in blocks]
    return {"role": "user", "content": content}


def assistant_message(*blocks: Any) -> dict:
    content = [text_block(b) if isinstance(b, str) else b for b in blocks]
    return {"role": "assistant", "content": content}


# ---------------------------------------------------------------------------
# Canonicalization (for cache keys)
# ---------------------------------------------------------------------------

def _canonicalize(obj: Any) -> Any:
    """Recursively replace raw bytes with a stable digest marker so the
    request canonical form is JSON-serializable and deterministic."""
    if isinstance(obj, (bytes, bytearray)):
        b = bytes(obj)
        return {
            "__bytes_sha256__": hashlib.sha256(b).hexdigest(),
            "__bytes_len__": len(b),
        }
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


def canonical_request(
    model: str,
    system: Optional[str],
    messages: list,
    decoding: dict,
    tags: Optional[dict],
) -> str:
    payload = {
        "model": model,
        "system": system,
        "messages": _canonicalize(messages),
        "decoding": _canonicalize(decoding),
        "tags": _canonicalize(tags or {}),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Thread-safe append-only JSONL cache
# ---------------------------------------------------------------------------

class JsonlCache:
    """Exact-match JSONL cache. One record per line:
    {"key", "request", "response", "ts"}.

    * append-only: existing lines are never rewritten
    * thread-safe within a process (single lock around index+file)
    * fail-closed: a key hit whose stored request differs from the current
      canonical request raises CacheCollisionError instead of returning data
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "bedrock_converse_cache.jsonl"
        self._lock = threading.Lock()
        self._index: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("cache: skipping corrupt line %d in %s", lineno, self.path)
                    continue
                key = rec.get("key")
                if key is None:
                    continue
                prev = self._index.get(key)
                if prev is not None and prev["request"] != rec["request"]:
                    raise CacheCollisionError(
                        f"cache file {self.path} line {lineno}: key {key} maps to "
                        "two different canonical requests (fail-closed)"
                    )
                # exact duplicates: keep first record
                self._index.setdefault(key, rec)

    def get(self, key: str, canonical: str) -> Optional[dict]:
        with self._lock:
            rec = self._index.get(key)
            if rec is None:
                return None
            if rec["request"] != canonical:
                raise CacheCollisionError(
                    f"cache key {key} collision: stored request differs from "
                    "current canonical request (fail-closed)"
                )
            return rec["response"]

    def put(self, key: str, canonical: str, response: dict) -> None:
        rec = {"key": key, "request": canonical, "response": response, "ts": time.time()}
        line = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        with self._lock:
            existing = self._index.get(key)
            if existing is not None:
                if existing["request"] != canonical:
                    raise CacheCollisionError(
                        f"cache key {key} collision on write (fail-closed)"
                    )
                return  # identical request already cached; append-only no-op
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._index[key] = rec


# ---------------------------------------------------------------------------
# Bedrock Converse client
# ---------------------------------------------------------------------------

class BedrockConverseClient:
    """Deterministic Bedrock Converse wrapper with caching and retries."""

    def __init__(
        self,
        region: str = REGION,
        cache_root: Path | str | None = None,
        verify_identity: bool = True,
        read_timeout: int = 600,
        connect_timeout: int = 30,
        max_attempts: int = 6,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
    ):
        if verify_identity:
            fail_fast_identity(region)
        self.region = region
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                read_timeout=read_timeout,
                connect_timeout=connect_timeout,
                retries={"max_attempts": 4, "mode": "adaptive"},
            ),
        )
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._cred_lock = threading.Lock()
        self._consecutive_cred_failures = 0
        self.cache: Optional[JsonlCache] = (
            JsonlCache(cache_root) if cache_root is not None else None
        )

    # -- credential failure accounting -----------------------------------
    def _note_cred_failure(self, code: str, exc: Exception) -> None:
        with self._cred_lock:
            self._consecutive_cred_failures += 1
            n = self._consecutive_cred_failures
        logger.warning("credential-shaped failure %d/%d (%s)", n,
                       MAX_CONSECUTIVE_CREDENTIAL_FAILURES, code)
        if n >= MAX_CONSECUTIVE_CREDENTIAL_FAILURES:
            raise CredentialFailureAbort(
                f"Aborting after {n} consecutive credential-shaped failures "
                f"(last: {code}: {exc}).\n{REMEDIATION_HINT}"
            ) from exc

    def _note_success(self) -> None:
        with self._cred_lock:
            self._consecutive_cred_failures = 0

    # -- main entry point --------------------------------------------------
    def converse(
        self,
        model_id: str,
        messages: list,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
        seed: Optional[int] = None,
        tags: Optional[dict] = None,
        use_cache: bool = True,
    ) -> dict:
        """Single Converse call.

        ``seed`` is *cache-key-only*: Anthropic models on Bedrock expose no
        seed parameter, so distinct seeds force distinct cache entries (for
        replicate runs) but are not sent to the API.

        Returns {"text", "stop_reason", "usage", "latency_ms", "cached",
        "model"}.
        """
        decoding = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop_sequences,
            "seed": seed,
        }
        canonical = canonical_request(model_id, system, messages, decoding, tags)
        key = cache_key(canonical)

        if use_cache and self.cache is not None:
            hit = self.cache.get(key, canonical)  # raises on collision
            if hit is not None:
                out = dict(hit)
                out["cached"] = True
                return out

        response = self._converse_with_retries(
            model_id, messages, system, max_tokens, temperature, top_p,
            stop_sequences,
        )
        if use_cache and self.cache is not None:
            self.cache.put(key, canonical, response)
        out = dict(response)
        out["cached"] = False
        return out

    # -- request execution --------------------------------------------------
    def _converse_with_retries(
        self,
        model_id: str,
        messages: list,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        stop_sequences: Optional[list[str]],
    ) -> dict:
        inference_config: dict[str, Any] = {"maxTokens": max_tokens}
        if not any(t in model_id for t in _NO_SAMPLING):
            inference_config["temperature"] = temperature
            if top_p is not None:
                inference_config["topP"] = top_p
        if stop_sequences:
            inference_config["stopSequences"] = stop_sequences

        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        if system:
            kwargs["system"] = [{"text": system}]

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            t0 = time.monotonic()
            try:
                resp = self._client.converse(**kwargs)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in CREDENTIAL_ERROR_CODES:
                    self._note_cred_failure(code, exc)  # may raise abort
                    last_exc = exc
                elif code in TRANSIENT_ERROR_CODES:
                    self._note_success()  # not credential-shaped
                    last_exc = exc
                else:
                    raise  # non-retryable (validation errors etc.)
                if attempt < self._max_attempts:
                    delay = min(
                        self._backoff_max,
                        self._backoff_base * (2 ** (attempt - 1)),
                    ) * (0.5 + random.random())
                    logger.warning(
                        "converse attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt, self._max_attempts, code, delay,
                    )
                    time.sleep(delay)
                continue

            latency_ms = int((time.monotonic() - t0) * 1000)
            self._note_success()
            content = resp.get("output", {}).get("message", {}).get("content", [])
            text = "".join(b.get("text", "") for b in content if "text" in b)
            return {
                "text": text,
                "stop_reason": resp.get("stopReason"),
                "usage": resp.get("usage", {}),
                "latency_ms": latency_ms,
                "model": model_id,
            }
        raise RuntimeError(
            f"converse failed after {self._max_attempts} attempts on "
            f"{model_id}: {last_exc}"
        ) from last_exc

    # -- convenience ---------------------------------------------------------
    def ask_text(self, model_id: str, prompt: str, **kw) -> dict:
        return self.converse(model_id, [user_message(prompt)], **kw)

    def ask_image_png(self, model_id: str, prompt: str, png_bytes: bytes, **kw) -> dict:
        msg = user_message(image_block_png(png_bytes), prompt)
        return self.converse(model_id, [msg], **kw)
