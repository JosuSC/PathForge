"""
backend/llm/client.py
---------------------
Cliente LLM multi-proveedor con cola circular de API keys.

v2 — Fixes:
[CL1] Groq, DeepSeek y Mistral añadidos a _PROVIDER_CALLERS y _DEFAULT_MODELS
      (todos usan API OpenAI-compatible). Validación temprana en _load_keys_from_env.
[CL2] complete() reescrito con lógica de loop clara: un intento por key disponible,
      reintentos solo en errores no-quota, sin condiciones de salida ambiguas.
[CL3] _get_next_available_key() hace get + rotate bajo un solo lock. Elimina
      race condition en entornos multi-thread (FastAPI con requests concurrentes).
[CL4] _call_gemini() con manejo explícito de ambos SDKs y sus configuraciones.
[CL5] sleep con jitter reducido (0.5-1.5s) y comentario explicativo.
[CL6] get_llm_client() cachea estado de error para evitar tormenta de reintentos.
"""

from __future__ import annotations

import os
import random
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────────────────────

class Provider(str, Enum):
    GEMINI     = "gemini"
    CLAUDE     = "claude"
    OPENAI     = "openai"
    OPENROUTER = "openrouter"
    GROQ       = "groq"       # FIX [CL1]: ahora soportado
    DEEPSEEK   = "deepseek"   # FIX [CL1]: ahora soportado
    MISTRAL    = "mistral"    # FIX [CL1]: ahora soportado


@dataclass
class LLMKey:
    """Representa una API key con su proveedor y estado de salud."""

    provider:       Provider
    key:            str
    failures:       int   = 0
    cooldown_until: float = 0.0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def mark_failure(self, cooldown_secs: float = 60.0) -> None:
        """Cooldown exponencial: 60s → 120s → 240s → max 600s."""
        self.failures += 1
        wait = min(cooldown_secs * (2 ** (self.failures - 1)), 600.0)
        self.cooldown_until = time.time() + wait
        logger.warning(
            f"Key [{self.provider.value}] en cooldown {wait:.0f}s "
            f"(fallo #{self.failures})"
        )

    def mark_success(self) -> None:
        self.failures       = 0
        self.cooldown_until = 0.0


# ──────────────────────────────────────────────────────────────
# Cargador de keys desde .env
# ──────────────────────────────────────────────────────────────

def _load_keys_from_env() -> list[LLMKey]:
    """
    Lee LLM_KEY_N del entorno. Formato: provider:api_key
    FIX [CL1]: valida que el proveedor tiene caller registrado.
    """
    keys: list[LLMKey] = []
    i = 1
    while True:
        raw = os.getenv(f"LLM_KEY_{i}")
        if not raw:
            break
        raw = raw.strip()
        if ":" not in raw:
            logger.warning(f"LLM_KEY_{i} formato inválido (esperado provider:key). Saltando.")
            i += 1
            continue

        provider_str, api_key = raw.split(":", 1)
        provider_str = provider_str.lower().strip()

        try:
            provider = Provider(provider_str)
        except ValueError:
            logger.warning(f"Proveedor desconocido en LLM_KEY_{i}: '{provider_str}'. Saltando.")
            i += 1
            continue

        if not api_key.strip():
            logger.warning(f"LLM_KEY_{i} tiene key vacía. Saltando.")
            i += 1
            continue

        # FIX [CL1]: validación temprana — avisar si no hay caller registrado
        if provider not in _PROVIDER_CALLERS:
            logger.warning(
                f"LLM_KEY_{i}: proveedor '{provider.value}' definido pero sin "
                f"adaptador registrado. Saltando."
            )
            i += 1
            continue

        keys.append(LLMKey(provider=provider, key=api_key.strip()))
        logger.info(f"Key cargada: LLM_KEY_{i} [{provider.value}]")
        i += 1

    # Fallback legacy: GEMINI_API_KEY
    if not keys:
        legacy = os.getenv("GEMINI_API_KEY", "").strip()
        if legacy:
            keys.append(LLMKey(provider=Provider.GEMINI, key=legacy))
            logger.info("Usando GEMINI_API_KEY como fallback (formato legacy)")

    return keys


# ──────────────────────────────────────────────────────────────
# Adaptadores por proveedor
# ──────────────────────────────────────────────────────────────

def _call_gemini(key: str, prompt: str, model: str) -> str:
    """
    FIX [CL4]: manejo explícito del nuevo SDK (google-genai) y del legacy
    (google-generativeai), con sus configuraciones correctas para cada uno.
    """
    try:
        # Nuevo SDK: google-genai >= 1.0
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        return (response.text or "").strip()

    except ImportError:
        pass
    except Exception as e:
        # El nuevo SDK está instalado pero falló — propagar el error
        raise

    # Fallback: legacy google-generativeai
    try:
        import google.generativeai as genai_legacy

        genai_legacy.configure(api_key=key)
        m = genai_legacy.GenerativeModel(
            model_name=model,
            generation_config=genai_legacy.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        response = m.generate_content(prompt)
        return response.text.strip()

    except ImportError:
        raise ImportError(
            "No se encontró ningún SDK de Gemini. "
            "Instala: pip install google-genai  (nuevo) "
            "o pip install google-generativeai  (legacy)"
        )


def _call_claude(key: str, prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _call_openai(key: str, prompt: str, model: str) -> str:
    import openai
    client = openai.OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _call_openrouter(key: str, prompt: str, model: str) -> str:
    import openai
    client = openai.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _call_groq(key: str, prompt: str, model: str) -> str:
    """FIX [CL1]: Groq — API OpenAI-compatible con base_url propio."""
    import openai
    client = openai.OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _call_deepseek(key: str, prompt: str, model: str) -> str:
    """FIX [CL1]: DeepSeek — API OpenAI-compatible."""
    import openai
    client = openai.OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _call_mistral(key: str, prompt: str, model: str) -> str:
    """FIX [CL1]: Mistral — API OpenAI-compatible."""
    import openai
    client = openai.OpenAI(api_key=key, base_url="https://api.mistral.ai/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# FIX [CL1]: Groq, DeepSeek y Mistral registrados
_PROVIDER_CALLERS: dict[Provider, Callable] = {
    Provider.GEMINI:     _call_gemini,
    Provider.CLAUDE:     _call_claude,
    Provider.OPENAI:     _call_openai,
    Provider.OPENROUTER: _call_openrouter,
    Provider.GROQ:       _call_groq,
    Provider.DEEPSEEK:   _call_deepseek,
    Provider.MISTRAL:    _call_mistral,
}

_DEFAULT_MODELS: dict[Provider, str] = {
    Provider.GEMINI:     "gemini-1.5-flash",
    Provider.CLAUDE:     "claude-haiku-4-5-20251001",
    Provider.OPENAI:     "gpt-4o-mini",
    Provider.OPENROUTER: "openrouter/auto",
    Provider.GROQ:       "llama-3.1-8b-instant",   # FIX [CL1]
    Provider.DEEPSEEK:   "deepseek-chat",           # FIX [CL1]
    Provider.MISTRAL:    "mistral-small-latest",    # FIX [CL1]
}

_QUOTA_ERROR_PATTERNS = (
    "quota", "rate", "limit", "429", "resource_exhausted",
    "insufficient_quota", "overloaded", "capacity", "too many",
)


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _QUOTA_ERROR_PATTERNS)


# ──────────────────────────────────────────────────────────────
# Cliente principal
# ──────────────────────────────────────────────────────────────

class LLMClient:
    """
    Cliente LLM multi-proveedor con cola circular de API keys.
    Thread-safe. Backoff exponencial. Soporta 7 proveedores.
    """

    def __init__(self, max_retries_per_key: int = 2) -> None:
        self._keys             = _load_keys_from_env()
        self._max_retries      = max_retries_per_key
        self._lock             = threading.Lock()
        self._current_idx: int = 0

        if not self._keys:
            raise EnvironmentError(
                "No se encontraron API keys válidas.\n"
                "Configura en .env:\n"
                "  LLM_KEY_1=gemini:tu_api_key\n"
                "  LLM_KEY_2=claude:sk-ant-...\n"
                "Proveedores soportados: "
                + ", ".join(p.value for p in _PROVIDER_CALLERS)
            )

        logger.info(
            f"LLMClient iniciado: {len(self._keys)} key(s) — "
            + ", ".join(f"[{k.provider.value}]" for k in self._keys)
        )

    # ── API pública ──────────────────────────────────────────────

    def complete(self, prompt: str) -> str:
        """
        Envía el prompt al LLM activo con rotación automática de keys.

        FIX [CL2]: loop claro — itera cada key disponible exactamente una vez.
        Si una key da error de quota → cooldown + rotar.
        Si una key da error no-quota → reintentar esa key (max_retries veces).
        Si todas fallan → RuntimeError.

        FIX [CL5]: sleep con jitter para errores transitorios (ejecutado en
        thread pool via run_in_executor en main_api, no bloquea event loop).
        """
        n_keys    = len(self._keys)
        keys_tried: set[int] = set()

        while len(keys_tried) < n_keys:
            key_obj, idx = self._get_next_available_key()

            if idx in keys_tried:
                # Dimos la vuelta completa sin encontrar una key fresca
                break
            keys_tried.add(idx)

            # Reintentos en esta key para errores no-quota
            for attempt in range(1, self._max_retries + 1):
                try:
                    result = self._call(key_obj, prompt)
                    key_obj.mark_success()
                    logger.debug(
                        f"LLM OK [{key_obj.provider.value}] "
                        f"key={idx} len={len(result)} attempt={attempt}"
                    )
                    return result

                except Exception as exc:
                    if _is_quota_error(exc):
                        logger.warning(
                            f"Quota/rate-limit key {idx} "
                            f"[{key_obj.provider.value}]: {exc}"
                        )
                        key_obj.mark_failure()
                        break  # salir del loop de reintentos, rotar key

                    # Error transitorio (red, timeout) → esperar y reintentar
                    jitter = random.uniform(0.5, 1.5)
                    logger.warning(
                        f"Error transitorio key {idx} "
                        f"[{key_obj.provider.value}] attempt {attempt}/{self._max_retries}: "
                        f"{exc}. Esperando {jitter:.1f}s..."
                    )
                    if attempt < self._max_retries:
                        # FIX [CL5]: sleep con jitter, documentado como blocking-ok
                        # (este método se llama desde run_in_executor en main_api)
                        time.sleep(jitter)
                    else:
                        key_obj.mark_failure(cooldown_secs=30.0)

        raise RuntimeError(
            f"Todas las {n_keys} API key(s) fallaron o están en cooldown. "
            "Revisa tu .env y los límites de cuota. "
            f"Estado: {self.status()}"
        )

    @property
    def active_provider(self) -> str:
        with self._lock:
            return self._keys[self._current_idx].provider.value

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def status(self) -> list[dict]:
        now = time.time()
        return [
            {
                "index":             i,
                "provider":          k.provider.value,
                "available":         k.is_available,
                "failures":          k.failures,
                "cooldown_remaining": max(0.0, round(k.cooldown_until - now, 1)),
            }
            for i, k in enumerate(self._keys)
        ]

    # ── Internals ────────────────────────────────────────────────

    def _get_next_available_key(self) -> tuple[LLMKey, int]:
        """
        FIX [CL3]: get + rotate bajo un SOLO lock para evitar race condition
        en entornos multi-thread (FastAPI con requests concurrentes).

        Si todas las keys están en cooldown, espera la que se libere antes.
        """
        with self._lock:
            n = len(self._keys)
            # Buscar la primera key disponible desde la posición actual
            for _ in range(n):
                key = self._keys[self._current_idx]
                idx = self._current_idx
                # Avanzar el índice para el próximo llamador (round-robin)
                self._current_idx = (self._current_idx + 1) % n
                if key.is_available:
                    return key, idx

            # Todas en cooldown → esperar la que se libere antes
            earliest = min(self._keys, key=lambda k: k.cooldown_until)
            wait     = max(0.0, earliest.cooldown_until - time.time())
            if wait > 0:
                logger.info(f"Todas las keys en cooldown. Esperando {wait:.1f}s...")

        # Sleep FUERA del lock para no bloquear otros threads
        if wait > 0:
            time.sleep(wait)

        with self._lock:
            idx = self._keys.index(earliest)
            self._current_idx = (idx + 1) % len(self._keys)
            return earliest, idx

    def _call(self, key_obj: LLMKey, prompt: str) -> str:
        """Llama al adaptador correcto según el proveedor."""
        model  = os.getenv(
            f"{key_obj.provider.value.upper()}_MODEL",
            _DEFAULT_MODELS[key_obj.provider],
        )
        caller = _PROVIDER_CALLERS[key_obj.provider]
        return caller(key_obj.key, prompt, model)


# ──────────────────────────────────────────────────────────────
# Singleton global (lazy, thread-safe)
# ──────────────────────────────────────────────────────────────

_client_instance: LLMClient | None  = None
_client_error:    Exception | None  = None
_client_lock                         = threading.Lock()


def get_llm_client() -> LLMClient:
    """
    Retorna la instancia singleton del cliente LLM.

    FIX [CL6]: cachea el estado de error — si la primera creación falla
    por falta de keys, los siguientes llamadores reciben el mismo error
    inmediatamente en lugar de reintentar N veces en paralelo.
    Para refrescar, llama a reset_llm_client() (útil en tests).
    """
    global _client_instance, _client_error

    with _client_lock:
        if _client_instance is not None:
            return _client_instance
        if _client_error is not None:
            raise _client_error

        try:
            _client_instance = LLMClient()
            return _client_instance
        except EnvironmentError as e:
            _client_error = e
            raise


def reset_llm_client() -> None:
    """Resetea el singleton (útil en tests o para recargar .env)."""
    global _client_instance, _client_error
    with _client_lock:
        _client_instance = None
        _client_error    = None
