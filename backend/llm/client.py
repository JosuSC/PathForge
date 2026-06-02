"""
backend/llm/client.py
---------------------
Cliente LLM multi-proveedor con cola circular de API keys.

Arquitectura:
    - Soporta Gemini, Claude (Anthropic) y OpenAI
    - Las keys se cargan desde .env como LLM_KEY_1..N
    - Cola circular: cuando una key falla por tokens/rate-limit,
      pasa automáticamente a la siguiente
    - Thread-safe con threading.Lock
    - Backoff exponencial por key antes de rotar

Formato .env:
    LLM_KEY_1=gemini:AIza...
    LLM_KEY_2=claude:sk-ant-...
    LLM_KEY_3=openai:sk-...
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
from collections import deque

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────────────────────

class Provider(str, Enum):
    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


@dataclass
class LLMKey:
    """Representa una API key con su proveedor y estado de salud."""

    provider: Provider
    key: str
    failures: int = 0
    cooldown_until: float = 0.0  # timestamp unix

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def mark_failure(self, cooldown_secs: float = 60.0) -> None:
        """Marca fallo y pone en cooldown exponencial."""
        self.failures += 1
        # 60s → 120s → 240s → max 600s
        wait = min(cooldown_secs * (2 ** (self.failures - 1)), 600.0)
        self.cooldown_until = time.time() + wait
        logger.warning(
            f"Key [{self.provider.value}] en cooldown por {wait:.0f}s "
            f"(fallo #{self.failures})"
        )

    def mark_success(self) -> None:
        self.failures = 0
        self.cooldown_until = 0.0


# ──────────────────────────────────────────────────────────────
# Cargador de keys desde .env
# ──────────────────────────────────────────────────────────────

def _load_keys_from_env() -> list[LLMKey]:
    """
    Lee todas las variables LLM_KEY_N del entorno.
    Formato: provider:api_key_value

    Ejemplo:
        LLM_KEY_1=gemini:AIza...
        LLM_KEY_2=claude:sk-ant-...
    """
    keys: list[LLMKey] = []
    i = 1
    while True:
        raw = os.getenv(f"LLM_KEY_{i}")
        if not raw:
            break
        raw = raw.strip()
        if ":" not in raw:
            logger.warning(f"LLM_KEY_{i} tiene formato inválido (esperado provider:key)")
            i += 1
            continue

        provider_str, api_key = raw.split(":", 1)
        try:
            provider = Provider(provider_str.lower().strip())
        except ValueError:
            logger.warning(f"Proveedor desconocido en LLM_KEY_{i}: '{provider_str}'. Saltando.")
            i += 1
            continue

        if not api_key.strip():
            logger.warning(f"LLM_KEY_{i} tiene key vacía. Saltando.")
            i += 1
            continue

        keys.append(LLMKey(provider=provider, key=api_key.strip()))
        logger.info(f"Key cargada: LLM_KEY_{i} [{provider.value}]")
        i += 1

    # Fallback: compatibilidad con .env antiguo que solo tiene GEMINI_API_KEY
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
    """Llama a Gemini priorizando google.genai y manteniendo fallback legacy."""
    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 1024,
            },
        )
        return (response.text or "").strip()
    except ImportError:
        # Fallback para entornos con la librería legacy instalada.
        import google.generativeai as genai

        genai.configure(api_key=key)
        m = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        response = m.generate_content(prompt)
        return response.text.strip()


def _call_claude(key: str, prompt: str, model: str) -> str:
    """Llama a la API de Anthropic Claude y retorna el texto."""
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _call_openai(key: str, prompt: str, model: str) -> str:
    """Llama a la API de OpenAI y retorna el texto."""
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
    """Llama a la API de OpenRouter (compatible con OpenAI) y retorna el texto."""
    import openai
    client = openai.OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# Mapa proveedor → función adaptadora
_PROVIDER_CALLERS: dict[Provider, Callable] = {
    Provider.GEMINI: _call_gemini,
    Provider.CLAUDE: _call_claude,
    Provider.OPENAI: _call_openai,
    Provider.OPENROUTER: _call_openrouter,
}

# Modelos por defecto
_DEFAULT_MODELS: dict[Provider, str] = {
    Provider.GEMINI: "gemini-1.5-flash",
    Provider.CLAUDE: "claude-haiku-4-5-20251001",
    Provider.OPENAI: "gpt-4o-mini",
    Provider.OPENROUTER: "openrouter/auto",
}

# Errores que indican quota/rate-limit (rotar key) vs errores fatales
_QUOTA_ERROR_PATTERNS = (
    "quota", "rate", "limit", "429", "resource_exhausted",
    "insufficient_quota", "overloaded", "capacity",
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

    - Rota automáticamente cuando una key se queda sin tokens
    - Thread-safe
    - Backoff exponencial por key en cooldown
    - Compatible con Gemini, Claude y OpenAI

    Uso:
        client = LLMClient()
        text = client.complete("tu prompt aquí")
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._keys: list[LLMKey] = _load_keys_from_env()
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._current_idx = 0  # índice actual en la cola circular

        if not self._keys:
            raise EnvironmentError(
                "No se encontraron API keys. "
                "Configura LLM_KEY_1, LLM_KEY_2... en tu .env\n"
                "Formato: LLM_KEY_1=gemini:tu_api_key"
            )

        logger.info(
            f"LLMClient iniciado con {len(self._keys)} key(s): "
            + ", ".join(f"[{k.provider.value}]" for k in self._keys)
        )

    # ── API pública ──────────────────────────────────────────────

    def complete(self, prompt: str) -> str:
        """
        Envía el prompt al LLM activo y retorna la respuesta.

        Rota automáticamente a la siguiente key en caso de error
        por quota/rate-limit. Lanza RuntimeError si todas fallan.
        """
        total_keys = len(self._keys)
        attempts_per_key = max(1, self._max_retries // total_keys)
        tried: set[int] = set()

        for _ in range(total_keys * attempts_per_key + total_keys):
            key_obj, idx = self._get_current_key()

            if idx in tried and len(tried) >= total_keys:
                break  # ya probamos todas las keys disponibles

            tried.add(idx)

            try:
                result = self._call(key_obj, prompt)
                key_obj.mark_success()
                logger.debug(
                    f"LLM OK [{key_obj.provider.value}] "
                    f"key_idx={idx} len={len(result)}"
                )
                return result

            except Exception as exc:
                if _is_quota_error(exc):
                    logger.warning(
                        f"Quota/rate-limit en key {idx} [{key_obj.provider.value}]: {exc}"
                    )
                    key_obj.mark_failure()
                    self._rotate()
                else:
                    # Error no relacionado con quota → reintentar misma key
                    logger.warning(
                        f"Error en key {idx} [{key_obj.provider.value}]: {exc}"
                    )
                    time.sleep(2)

        raise RuntimeError(
            f"Todas las {total_keys} API key(s) fallaron. "
            "Revisa tu .env y los límites de cuota."
        )

    @property
    def active_provider(self) -> str:
        """Nombre del proveedor actualmente activo."""
        with self._lock:
            return self._keys[self._current_idx].provider.value

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def status(self) -> list[dict]:
        """Retorna el estado de todas las keys (para debugging)."""
        now = time.time()
        return [
            {
                "index": i,
                "provider": k.provider.value,
                "available": k.is_available,
                "failures": k.failures,
                "cooldown_remaining": max(0.0, round(k.cooldown_until - now, 1)),
            }
            for i, k in enumerate(self._keys)
        ]

    # ── Internals ────────────────────────────────────────────────

    def _get_current_key(self) -> tuple[LLMKey, int]:
        """Retorna la key actual. Si está en cooldown, busca la siguiente disponible."""
        with self._lock:
            start = self._current_idx
            for _ in range(len(self._keys)):
                key = self._keys[self._current_idx]
                if key.is_available:
                    return key, self._current_idx
                # Esta key está en cooldown → saltar
                self._current_idx = (self._current_idx + 1) % len(self._keys)

            # Todas en cooldown → esperar la que se libere antes
            earliest_key = min(self._keys, key=lambda k: k.cooldown_until)
            wait = max(0.0, earliest_key.cooldown_until - time.time())
            if wait > 0:
                logger.info(f"Todas las keys en cooldown. Esperando {wait:.1f}s...")
                time.sleep(wait)
            return earliest_key, self._keys.index(earliest_key)

    def _rotate(self) -> None:
        """Avanza al siguiente índice en la cola circular."""
        with self._lock:
            self._current_idx = (self._current_idx + 1) % len(self._keys)
            logger.info(
                f"Rotando a key {self._current_idx} "
                f"[{self._keys[self._current_idx].provider.value}]"
            )

    def _call(self, key_obj: LLMKey, prompt: str) -> str:
        """Llama al adaptador correcto según el proveedor."""
        model = os.getenv(
            f"{key_obj.provider.value.upper()}_MODEL",
            _DEFAULT_MODELS[key_obj.provider],
        )
        caller = _PROVIDER_CALLERS[key_obj.provider]
        return caller(key_obj.key, prompt, model)


# ──────────────────────────────────────────────────────────────
# Singleton global (lazy)
# ──────────────────────────────────────────────────────────────
_client_instance: LLMClient | None = None
_client_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """
    Retorna la instancia singleton del cliente LLM.
    Thread-safe, se crea solo la primera vez que se llama.
    """
    global _client_instance
    with _client_lock:
        if _client_instance is None:
            _client_instance = LLMClient()
        return _client_instance