"""
llm/client.py
-------------
Cliente Gemini con reintentos, timeout y manejo de errores.

Patrón: Adapter — abstrae la API de Gemini detrás de una interfaz
simple para que el resto del sistema sea agnóstico al proveedor LLM.
"""

from __future__ import annotations

import os
import time
from typing import Any

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class GeminiClient:
    """
    Wrapper sobre google-generativeai con:
    - Configuración centralizada
    - Reintentos con backoff exponencial
    - Logging automático de llamadas
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        max_retries: int = 3,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY no encontrada. "
                "Agrégala al archivo .env"
            )

        genai.configure(api_key=api_key)

        self._model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._max_retries = max_retries
        self._generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=self._generation_config,
        )

        logger.info(f"GeminiClient inicializado con modelo: {self._model_name}")

    def complete(self, prompt: str) -> str:
        """
        Envía un prompt y retorna la respuesta como string.
        Reintenta hasta max_retries veces con backoff exponencial.

        Args:
            prompt: Texto del prompt.

        Returns:
            Respuesta del modelo como string limpio.

        Raises:
            RuntimeError: Si todos los reintentos fallan.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    f"Gemini request | attempt={attempt} | "
                    f"prompt_len={len(prompt)}"
                )
                response = self._model.generate_content(prompt)
                result = response.text.strip()
                logger.debug(f"Gemini response | len={len(result)}")
                return result

            except Exception as exc:
                wait = 2 ** attempt  # backoff: 2s, 4s, 8s
                logger.warning(
                    f"Gemini error (attempt {attempt}/{self._max_retries}): "
                    f"{exc}. Reintentando en {wait}s..."
                )
                if attempt == self._max_retries:
                    raise RuntimeError(
                        f"Gemini falló tras {self._max_retries} intentos: {exc}"
                    ) from exc
                time.sleep(wait)

        return ""  # nunca llega aquí, satisface el type checker
