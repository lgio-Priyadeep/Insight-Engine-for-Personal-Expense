"""
chatbot_context_loader.py — Context path resolution and file loading for the LLM chatbot.

Pure functions — no module-level filesystem access, no side effects.
All functions are fully testable without a real filesystem (use tmp_path).
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import config

# Resolved once at import — matches config and env var INSIGHT_LLM_EXPORT_DIR.
SAFE_BASE_DIR: str = getattr(config, "LLM_EXPORT_OUTPUT_DIR", "output/llm_context")

# Only schema_version "1.0" is currently accepted.
ACCEPTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})

_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def hash_token(token: str) -> str:
    """
    Return the first 16 hex characters of SHA-256(token).
    Used as the per-user subdirectory name under SAFE_BASE_DIR.

    Do NOT use hash_utils.stable_hash(): it is deprecated, emits
    DeprecationWarning on every call, returns only 12 chars, and its
    docstring explicitly forbids this use case.
    """
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def resolve_context_path(run_id: str | None, user_hash: str) -> str:
    """
    Resolve the absolute path to a user's LLM context JSON file.

    If run_id is given:
        - Validates format against ^[0-9a-f]{12}$ (raises ValueError on mismatch).
        - Builds candidate path and calls os.path.realpath().
        - Asserts realpath starts with realpath(SAFE_BASE_DIR) + os.sep (path traversal guard).

    If run_id is None (auto-latest):
        - Scans SAFE_BASE_DIR/<user_hash>/ for *_llm_context.json files.
        - Returns the one with the highest mtime.
        - Raises FileNotFoundError if the directory is missing or empty.

    Args:
        run_id:    12-char lowercase hex string, or None for auto-latest.
        user_hash: 16-char hex string from hash_token().

    Returns:
        Absolute realpath to the context JSON file.

    Raises:
        ValueError:        run_id present but format invalid, or path traversal detected.
        FileNotFoundError: run_id absent and no context files found for user,
                           OR run_id present but the resolved file does not exist.
    """
    user_dir = os.path.join(SAFE_BASE_DIR, user_hash)

    if run_id is not None:
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(
                f"Invalid run_id format: '{run_id}'. Must match ^[0-9a-f]{{12}}$."
            )
        candidate = os.path.join(user_dir, f"{run_id}_llm_context.json")
        real_candidate = os.path.realpath(candidate)
        real_base = os.path.realpath(SAFE_BASE_DIR)
        if not real_candidate.startswith(real_base + os.sep):
            raise ValueError(
                f"Path traversal detected for run_id '{run_id}'."
            )
        if not os.path.isfile(real_candidate):
            raise FileNotFoundError(
                f"No context file found for run_id '{run_id}'."
            )
        return real_candidate

    # Auto-latest: scan user subdirectory only (never falls back to flat dir).
    if not os.path.isdir(user_dir):
        raise FileNotFoundError(
            "No context directory found for this user. Run analysis first."
        )

    candidates = [
        f for f in os.listdir(user_dir)
        if f.endswith("_llm_context.json")
    ]
    if not candidates:
        raise FileNotFoundError(
            "No exported analysis found for this user. Run analysis first."
        )

    candidates.sort(
        key=lambda f: os.path.getmtime(os.path.join(user_dir, f)),
        reverse=True,
    )
    return os.path.realpath(os.path.join(user_dir, candidates[0]))


def load_context(path: str) -> dict:
    """
    Load and validate a LLM context JSON file from an absolute path.

    Checks (in order):
      1. os.path.exists() — raises FileNotFoundError if absent.
      2. os.path.getsize() > 0 — raises ValueError for empty file.
      3. json.load() — raises ValueError (wrapping JSONDecodeError) for malformed JSON.
      4. schema_version in ACCEPTED_SCHEMA_VERSIONS — raises ValueError for unknown version.

    Args:
        path: Absolute path to context file (from resolve_context_path).

    Returns:
        Parsed context dict.

    Raises:
        FileNotFoundError: file absent.
        ValueError:        empty file, bad JSON, or unsupported schema_version.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Context file not found: {path}")

    if os.path.getsize(path) == 0:
        raise ValueError(f"Context file is empty: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            context = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Context file is not valid JSON: {exc}") from exc

    version = context.get("schema_version", "")
    if version not in ACCEPTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version '{version}'. "
            f"Accepted: {sorted(ACCEPTED_SCHEMA_VERSIONS)}"
        )

    return context


__all__ = ["hash_token", "resolve_context_path", "load_context", "SAFE_BASE_DIR"]
