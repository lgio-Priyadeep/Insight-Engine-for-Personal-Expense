"""
chatbot_engine.py — LLM singleton and streaming inference for the chatbot.

B3 FIX: generate_response() uses stream=True with a token-by-token elapsed-time
check. It breaks the loop at the first token boundary after the deadline — the
only point where llama.cpp yields control back to Python. This releases _lock
immediately on timeout.

DO NOT replace stream=True with concurrent.futures.wait() or threading.Timer:
neither can interrupt C-level code. The underlying thread would continue holding
_lock for up to 10+ minutes after the handler returns 504, blocking all
subsequent chat requests (silent DoS).
"""
from __future__ import annotations

import logging
import threading
import time

import config

logger = logging.getLogger(__name__)

_lock: threading.Lock = threading.Lock()
_llm_instance = None       # Llama | None — module-level singleton


class ModelNotReadyError(RuntimeError):
    """Model file missing or failed to load."""


class InferenceTimeoutError(RuntimeError):
    """Inference exceeded the configured timeout."""


def _get_llm_instance():
    """
    Lazy-load and return the Llama singleton.
    Must be called while _lock is already held by the caller.

    Raises:
        ModelNotReadyError: if model file is missing or Llama() raises.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    import os
    model_path: str = getattr(
        config, "CHATBOT_MODEL_PATH",
        "models/llama-3.2-3b-instruct-q4_k_m.gguf"
    )

    if not os.path.isfile(model_path):
        raise ModelNotReadyError(f"Model file not found: {model_path}")

    try:
        from llama_cpp import Llama
        _llm_instance = Llama(
            model_path=model_path,
            n_ctx=getattr(config, "CHATBOT_N_CTX", 4096),
            n_threads=getattr(config, "CHATBOT_N_THREADS", 4),
            verbose=False,
        )
        logger.info("llm_model_loaded", extra={"model_path": model_path})
    except Exception as exc:
        raise ModelNotReadyError(f"Failed to load model '{model_path}': {exc}") from exc

    return _llm_instance


def generate_response(messages: list[dict], timeout_s: int | None = None) -> str:
    """
    Generate a chat completion. Synchronous — MUST be called via run_in_executor.

    B3 FIX — stream=True token-by-token timeout:
      Uses llm.create_chat_completion(..., stream=True) to receive an iterator
      of token chunks. On each chunk, checks time.monotonic() - start > timeout_s.
      If True: breaks the loop and raises InferenceTimeoutError.
      The C-level loop exits at this token boundary; _lock is released by the
      context manager immediately — not after inference finishes.

    Caller pattern (in async handler):
      answer = await asyncio.get_running_loop().run_in_executor(
          None, generate_response, messages
      )

    Args:
        messages:  List of chat message dicts [{"role": ..., "content": ...}].
        timeout_s: Override timeout in seconds. Defaults to config.CHATBOT_TIMEOUT_S.

    Returns:
        Assembled response string.

    Raises:
        ModelNotReadyError:    model file missing or failed to load.
        InferenceTimeoutError: generation exceeded timeout_s.
    """
    if timeout_s is None:
        timeout_s = getattr(config, "CHATBOT_TIMEOUT_S", 400)
    max_tokens: int = getattr(config, "CHATBOT_MAX_RESPONSE_TOKENS", 512)

    with _lock:
        llm = _get_llm_instance()
        start = time.monotonic()
        tokens: list[str] = []

        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            stream=True,      # B3 fix: returns iterator; Python regains control per token
        )

        for chunk in stream:
            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                logger.warning(
                    "inference_timeout",
                    extra={"elapsed_s": round(elapsed, 1), "timeout_s": timeout_s},
                )
                raise InferenceTimeoutError(
                    f"LLM did not respond within {timeout_s}s"
                )
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                tokens.append(delta)

        return "".join(tokens)


__all__ = ["generate_response", "ModelNotReadyError", "InferenceTimeoutError"]
