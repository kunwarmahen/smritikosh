"""
LLM Adapter — unified interface over any provider via LiteLLM.

Supported providers:
    claude    → Anthropic Claude  (claude-haiku-4-5-20251001, claude-sonnet-4-6 …)
    openai    → OpenAI            (gpt-4o, gpt-4o-mini …)
    gemini    → Google Gemini     (gemini/gemini-1.5-pro …)
    ollama    → local Ollama      (ollama/qwen2.5:7b, ollama/llama3 …)
    vllm      → local vLLM        (openai/qwen — served at LLM_BASE_URL)
    llamacpp  → local llama.cpp   (openai/<model> — served at LLM_BASE_URL, default http://localhost:8080/v1)

LiteLLM translates all of these to one consistent interface, so the rest of
the codebase never needs to know which provider is active.
"""

import json
import logging
from typing import Any

import litellm
from tenacity import before_sleep_log, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from smritikosh.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

# Suppress litellm's verbose startup banner
litellm.suppress_debug_info = True


class LLMAdapter:
    """
    Handles all LLM interactions for Smritikosh:
      - complete()           → free-form chat completion
      - extract_structured() → returns a validated Python dict (JSON mode)
      - embed()              → returns a float vector for a piece of text
    """

    def __init__(self, cfg: Settings = default_settings) -> None:
        self._cfg = cfg
        self._chat_model = self._resolve_chat_model(cfg)
        self._embed_model = self._resolve_embed_model(cfg)
        logger.info(
            "LLMAdapter initialised",
            extra={
                "chat_provider": cfg.llm_provider,
                "chat_model": self._chat_model,
                "embed_provider": cfg.embedding_provider,
                "embed_model": self._embed_model,
                "embed_dimensions": cfg.embedding_dimensions,
            },
        )

    # ── Public interface ───────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return the response text."""
        if self._cfg.llm_max_tokens is not None:
            kwargs.setdefault("max_tokens", self._cfg.llm_max_tokens)
        # Disable thinking tokens for all Ollama calls — thinking consumes the token
        # budget and leaves content empty when max_tokens is low (e.g. intent classifier).
        if self._cfg.llm_provider.lower() == "ollama":
            kwargs.setdefault("extra_body", {"think": False})
        logger.debug("LLM complete: model=%s messages=%d", self._chat_model, len(messages))
        response = await litellm.acompletion(
            model=self._chat_model,
            messages=messages,
            temperature=temperature,
            api_key=self._cfg.llm_api_key,
            api_base=self._cfg.llm_base_url,
            **kwargs,
        )
        msg = response.choices[0].message
        content = msg.content
        if not content:
            # Some thinking models (e.g. gemma4 via ollama_chat) put their
            # output in reasoning_content and leave content None/empty.
            reasoning = getattr(msg, "reasoning_content", None)
            logger.warning(
                "LLM returned empty content (model=%s) reasoning_content_len=%d",
                self._chat_model,
                len(reasoning) if reasoning else 0,
            )
            raise RuntimeError(
                f"LLM returned empty content (model={self._chat_model})"
            )
        logger.debug("LLM response: %d chars", len(content))
        return content

    # ValueError means the LLM returned bad JSON — deterministic failure, no point retrying.
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type(ValueError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def extract_structured(
        self,
        prompt: str,
        schema_description: str,
        example_output: dict,
    ) -> dict:
        """
        Ask the LLM to return a JSON object matching the given schema.

        Uses a system prompt that instructs strict JSON output — compatible
        with providers that don't natively support JSON mode (Ollama, vLLM).

        Args:
            prompt:             The user-facing instruction.
            schema_description: Plain-English description of the expected fields.
            example_output:     A concrete example dict to show the LLM.

        Returns:
            Parsed dict. Raises ValueError if the LLM returns malformed JSON.
        """
        system = (
            "You are a precise data extractor. "
            "Respond ONLY with a valid JSON object — no markdown fences, no explanation. "
            f"Schema: {schema_description}. "
            f"Example: {json.dumps(example_output)}"
        )
        # response_format={"type": "json_object"} enables native JSON mode for
        # OpenAI/Claude/Gemini. For Ollama it maps to format="json" which causes
        # some models to hang — Ollama relies on the system prompt instruction instead.
        extra: dict = {}
        if self._cfg.llm_provider.lower() not in ("ollama", "llamacpp"):
            extra["response_format"] = {"type": "json_object"}
        raw = await self.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # deterministic extraction
            **extra,
        )
        return self._parse_json(raw)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def embed(self, text: str) -> list[float]:
        """Generate a float vector for the given text using the configured embedding model."""
        logger.debug("Embedding: model=%s text_len=%d", self._embed_model, len(text))
        if self._cfg.embedding_provider.lower() == "llamacpp":
            return await self._embed_llamacpp(text)
        response = await litellm.aembedding(
            model=self._embed_model,
            input=text,
            api_key=self._cfg.embedding_api_key,
            api_base=self._cfg.embedding_base_url,
        )
        return response.data[0]["embedding"]

    async def _embed_llamacpp(self, text: str) -> list[float]:
        """Call llama.cpp native /embedding endpoint — OAI-compatible /v1/embeddings
        fails on chat models due to chat-template processing. The native endpoint
        bypasses the template entirely and returns embeddings directly."""
        import httpx
        base = (self._cfg.embedding_base_url or "http://localhost:8080/v1").rstrip("/")
        # Strip /v1 suffix to reach the native endpoint
        native_url = base.removesuffix("/v1") + "/embedding"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(native_url, json={"content": text})
            resp.raise_for_status()
        data = resp.json()
        # Response: [{"index": 0, "embedding": [[...floats...]]}]
        embedding = data[0]["embedding"]
        return embedding[0] if isinstance(embedding[0], list) else embedding

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def describe_image(self, image_bytes: bytes, filename: str, prompt: str) -> str:
        """
        Describe an image using a vision-capable model.

        Encodes the image as base64 and sends it alongside the prompt using
        the OpenAI multimodal message format, which LiteLLM translates for
        Claude, Gemini, and local providers automatically.

        Args:
            image_bytes: Raw image file bytes.
            filename:    Original filename — used to infer the MIME type.
            prompt:      Task-specific instruction (e.g. receipt vs screenshot prompt).

        Returns:
            A plain-text description of the image content.
        """
        import base64

        ext_to_mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mime_type = ext_to_mime.get(ext, "image/jpeg")

        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"

        vision_model = self._resolve_vision_model(self._cfg)
        api_key = self._cfg.vision_api_key or self._cfg.llm_api_key
        api_base = self._cfg.vision_base_url or self._cfg.llm_base_url

        logger.debug(
            "Describing image: model=%s filename=%s size=%d",
            vision_model,
            filename,
            len(image_bytes),
        )

        response = await litellm.acompletion(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.2,
            api_key=api_key,
            api_base=api_base,
        )
        content = response.choices[0].message.content or ""
        logger.debug("Image description complete: %d chars", len(content))
        return content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """
        Transcribe audio file using Whisper.

        Supports two providers:
        - OpenAI (cloud): requires WHISPER_API_KEY or EMBEDDING_API_KEY
        - Local (self-hosted): requires WHISPER_BASE_URL (ollama/vllm/llamacpp)

        Set WHISPER_PROVIDER env var to "openai" or "local".
        """
        import io

        provider = self._cfg.whisper_provider.lower()

        if provider == "openai":
            return await self._transcribe_openai(audio_bytes, filename)
        elif provider == "local":
            return await self._transcribe_local(audio_bytes, filename)
        else:
            raise ValueError(
                f"Unknown whisper_provider: {provider}. Must be 'openai' or 'local'."
            )

    async def _transcribe_openai(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe using OpenAI Whisper API via litellm."""
        import io

        api_key = self._cfg.whisper_api_key or self._cfg.embedding_api_key
        if not api_key:
            raise ValueError(
                "OpenAI Whisper requires WHISPER_API_KEY or EMBEDDING_API_KEY to be set"
            )

        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = filename
        logger.debug(
            "Transcribing via OpenAI: model=%s filename=%s size=%d",
            self._cfg.whisper_model,
            filename,
            len(audio_bytes),
        )

        response = await litellm.atranscription(
            model=self._cfg.whisper_model,
            file=file_obj,
            api_key=api_key,
        )
        text = response.text if hasattr(response, "text") else str(response)
        logger.debug("OpenAI transcription complete: length=%d chars", len(text))
        return text

    async def _transcribe_local(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe using local Whisper (ollama/vllm/llamacpp) via litellm."""
        import io

        base_url = self._cfg.whisper_base_url
        if not base_url:
            raise ValueError(
                "Local Whisper requires WHISPER_BASE_URL to be set "
                "(e.g., http://localhost:8000 for ollama)"
            )

        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = filename
        logger.debug(
            "Transcribing via local Whisper: model=%s base_url=%s filename=%s",
            self._cfg.whisper_model,
            base_url,
            filename,
        )

        # For local providers, use litellm's routing via the model prefix
        model_string = f"openai/{self._cfg.whisper_model}"

        response = await litellm.atranscription(
            model=model_string,
            file=file_obj,
            api_base=base_url,
        )
        text = response.text if hasattr(response, "text") else str(response)
        logger.debug("Local transcription complete: length=%d chars", len(text))
        return text

    # ── Model string resolution ────────────────────────────────────────────

    @staticmethod
    def _resolve_chat_model(cfg: Settings) -> str:
        """
        Map provider + model name to the LiteLLM model string format.

        LiteLLM routing rules:
            claude    → model string as-is          (e.g. "claude-haiku-4-5-20251001")
            openai    → model string as-is          (e.g. "gpt-4o")
            gemini    → "gemini/<model>"            (e.g. "gemini/gemini-1.5-pro")
            ollama    → "ollama_chat/<model>"        (e.g. "ollama_chat/gemma4:26b")
            vllm      → "openai/<model>" + base_url (vLLM mimics OpenAI API)
            llamacpp  → "openai/<model>" + base_url (llama-server mimics OpenAI API)
        """
        provider = cfg.llm_provider.lower()
        model = cfg.llm_model

        if provider == "gemini" and not model.startswith("gemini/"):
            return f"gemini/{model}"
        if provider == "ollama" and not model.startswith("ollama_chat/"):
            # ollama_chat/ routes to /api/chat which correctly handles thinking-model
            # responses (qwen3.5, deepseek-r1, etc.) where reasoning tokens are
            # returned in a separate field. ollama/ uses /api/generate which drops them.
            return f"ollama_chat/{model}"
        if provider in ("vllm", "llamacpp") and not model.startswith("openai/"):
            return f"openai/{model}"
        # claude and openai use the model name directly
        return model

    @staticmethod
    def _resolve_vision_model(cfg: Settings) -> str:
        """Same routing logic for the vision model."""
        provider = cfg.vision_provider.lower()
        model = cfg.vision_model

        if provider == "gemini" and not model.startswith("gemini/"):
            return f"gemini/{model}"
        if provider == "ollama" and not model.startswith("ollama_chat/"):
            return f"ollama_chat/{model}"
        if provider in ("vllm", "llamacpp") and not model.startswith("openai/"):
            return f"openai/{model}"
        return model

    @staticmethod
    def _resolve_embed_model(cfg: Settings) -> str:
        """Same routing logic for the embedding model."""
        provider = cfg.embedding_provider.lower()
        model = cfg.embedding_model

        if provider == "gemini" and not model.startswith("gemini/"):
            return f"gemini/{model}"
        if provider == "ollama" and not model.startswith("ollama/"):
            return f"ollama/{model}"
        if provider in ("vllm", "llamacpp") and not model.startswith("openai/"):
            return f"openai/{model}"
        return model

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Strip thinking tokens and markdown fences, then parse JSON."""
        import re
        text = raw.strip()
        # Strip <think>...</think> blocks emitted by reasoning models (e.g. Qwen3.5)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            # Remove ```json ... ``` wrapper if present
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned invalid JSON (len=%d): %.200s", len(raw), raw)
            raise ValueError(f"LLM returned invalid JSON: {raw!r}") from exc
