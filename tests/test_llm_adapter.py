"""
Tests for LLMAdapter.

How to run:
    # Unit tests only (no real LLM calls, no API keys needed):
    pytest tests/test_llm_adapter.py -v

    # Live integration test against Claude (needs ANTHROPIC_API_KEY / LLM_API_KEY):
    pytest tests/test_llm_adapter.py -v -m live

    # Live integration test against Ollama (needs Ollama running locally):
    pytest tests/test_llm_adapter.py -v -m ollama

Test strategy:
    - Unit tests mock litellm so they run offline and never hit real APIs.
    - Live tests (marked @pytest.mark.live / @pytest.mark.ollama) are skipped
      by default and only run when explicitly requested.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.config import Settings
from smritikosh.llm.adapter import LLMAdapter


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_adapter(**overrides) -> LLMAdapter:
    """Build an LLMAdapter with test-safe settings (no real keys needed)."""
    cfg = Settings(
        llm_provider=overrides.get("llm_provider", "claude"),
        llm_model=overrides.get("llm_model", "claude-haiku-4-5-20251001"),
        llm_api_key=overrides.get("llm_api_key", "test-key"),
        llm_base_url=overrides.get("llm_base_url", None),
        embedding_provider=overrides.get("embedding_provider", "openai"),
        embedding_model=overrides.get("embedding_model", "text-embedding-3-small"),
        embedding_api_key=overrides.get("embedding_api_key", "test-key"),
        embedding_base_url=overrides.get("embedding_base_url", None),
        embedding_dimensions=1536,
    )
    return LLMAdapter(cfg)


def fake_completion_response(content: str) -> MagicMock:
    """Builds a litellm-shaped response object."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def fake_embedding_response(vector: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.data = [{"embedding": vector}]
    return resp


# ── Model string resolution ────────────────────────────────────────────────────

class TestModelResolution:
    def test_claude_model_unchanged(self):
        adapter = make_adapter(llm_provider="claude", llm_model="claude-haiku-4-5-20251001")
        assert adapter._chat_model == "claude-haiku-4-5-20251001"

    def test_openai_model_unchanged(self):
        adapter = make_adapter(llm_provider="openai", llm_model="gpt-4o")
        assert adapter._chat_model == "gpt-4o"

    def test_gemini_prefixed(self):
        adapter = make_adapter(llm_provider="gemini", llm_model="gemini-1.5-pro")
        assert adapter._chat_model == "gemini/gemini-1.5-pro"

    def test_gemini_not_double_prefixed(self):
        """If model already has 'gemini/' prefix, don't add it again."""
        adapter = make_adapter(llm_provider="gemini", llm_model="gemini/gemini-1.5-pro")
        assert adapter._chat_model == "gemini/gemini-1.5-pro"

    def test_ollama_prefixed(self):
        adapter = make_adapter(llm_provider="ollama", llm_model="qwen2.5:7b")
        assert adapter._chat_model == "ollama/qwen2.5:7b"

    def test_vllm_prefixed(self):
        adapter = make_adapter(llm_provider="vllm", llm_model="qwen2.5-72b")
        assert adapter._chat_model == "openai/qwen2.5-72b"

    def test_embed_model_ollama_prefixed(self):
        adapter = make_adapter(embedding_provider="ollama", embedding_model="nomic-embed-text")
        assert adapter._embed_model == "ollama/nomic-embed-text"


# ── complete() ────────────────────────────────────────────────────────────────

class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_content(self):
        adapter = make_adapter()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = fake_completion_response("Hello from the brain")
            result = await adapter.complete([{"role": "user", "content": "Hi"}])

        assert result == "Hello from the brain"

    @pytest.mark.asyncio
    async def test_complete_passes_correct_model_and_key(self):
        adapter = make_adapter(llm_provider="claude", llm_api_key="sk-ant-test")
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = fake_completion_response("ok")
            await adapter.complete([{"role": "user", "content": "test"}])

        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["api_key"] == "sk-ant-test"

    @pytest.mark.asyncio
    async def test_complete_passes_base_url_for_ollama(self):
        adapter = make_adapter(
            llm_provider="ollama",
            llm_model="qwen2.5:7b",
            llm_base_url="http://localhost:11434",
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = fake_completion_response("ok")
            await adapter.complete([{"role": "user", "content": "test"}])

        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["api_base"] == "http://localhost:11434"
        assert call_kwargs["model"] == "ollama/qwen2.5:7b"


# ── extract_structured() ──────────────────────────────────────────────────────

class TestExtractStructured:
    @pytest.mark.asyncio
    async def test_returns_parsed_dict(self):
        adapter = make_adapter()
        payload = {"facts": [{"type": "preference", "key": "color", "value": "green"}]}
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = fake_completion_response(json.dumps(payload))
            result = await adapter.extract_structured(
                prompt="Extract facts from: user likes green",
                schema_description="facts: list of {type, key, value}",
                example_output={"facts": []},
            )

        assert result == payload

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        adapter = make_adapter()
        raw = "```json\n{\"key\": \"value\"}\n```"
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = fake_completion_response(raw)
            result = await adapter.extract_structured(
                prompt="test",
                schema_description="key: string",
                example_output={"key": "example"},
            )

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        adapter = make_adapter()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = fake_completion_response("this is not json")
            with pytest.raises(ValueError, match="invalid JSON"):
                await adapter.extract_structured(
                    prompt="test",
                    schema_description="any",
                    example_output={},
                )

    @pytest.mark.asyncio
    async def test_uses_zero_temperature_for_determinism(self):
        adapter = make_adapter()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = fake_completion_response("{}")
            await adapter.extract_structured("prompt", "schema", {})

        assert mock_llm.call_args.kwargs["temperature"] == 0.0


# ── embed() ───────────────────────────────────────────────────────────────────

class TestEmbed:
    @pytest.mark.asyncio
    async def test_returns_float_vector(self):
        adapter = make_adapter()
        vector = [0.1, 0.2, 0.3]
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = fake_embedding_response(vector)
            result = await adapter.embed("some text about memory")

        assert result == vector

    @pytest.mark.asyncio
    async def test_embed_passes_correct_model(self):
        adapter = make_adapter(embedding_provider="openai", embedding_model="text-embedding-3-small")
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = fake_embedding_response([0.0])
            await adapter.embed("test")

        assert mock_embed.call_args.kwargs["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_passes_base_url_for_ollama(self):
        adapter = make_adapter(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            embedding_base_url="http://localhost:11434",
        )
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = fake_embedding_response([0.0])
            await adapter.embed("test")

        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["model"] == "ollama/nomic-embed-text"
        assert call_kwargs["api_base"] == "http://localhost:11434"


# ── Live integration tests (skipped by default) ───────────────────────────────

@pytest.mark.live
class TestLiveIntegration:
    """
    Run with: pytest tests/test_llm_adapter.py -v -m live

    Requires real API keys in .env.
    """

    @pytest.mark.asyncio
    async def test_claude_complete(self):
        from smritikosh.config import settings
        adapter = LLMAdapter(settings)
        result = await adapter.complete([
            {"role": "user", "content": "Reply with exactly: MEMORY_OK"}
        ])
        assert "MEMORY_OK" in result

    @pytest.mark.asyncio
    async def test_claude_extract_structured(self):
        from smritikosh.config import settings
        adapter = LLMAdapter(settings)
        result = await adapter.extract_structured(
            prompt="User said: I prefer dark mode and I work in AI.",
            schema_description="facts: list of {type: preference|interest|role, value: string}",
            example_output={"facts": [{"type": "preference", "value": "dark mode"}]},
        )
        assert "facts" in result
        assert isinstance(result["facts"], list)

    @pytest.mark.asyncio
    async def test_claude_embed_returns_vector(self):
        from smritikosh.config import settings
        adapter = LLMAdapter(settings)
        vector = await adapter.embed("human memory and the hippocampus")
        assert len(vector) > 100  # embedding vectors are large


@pytest.mark.ollama
class TestOllamaIntegration:
    """
    Run with: pytest tests/test_llm_adapter.py -v -m ollama

    Requires Ollama running locally:
        ollama serve
        ollama pull qwen2.5:7b
        ollama pull nomic-embed-text
    """

    @pytest.mark.asyncio
    async def test_ollama_complete(self):
        adapter = make_adapter(
            llm_provider="ollama",
            llm_model="qwen2.5:7b",
            llm_base_url="http://localhost:11434",
            llm_api_key=None,
        )
        result = await adapter.complete([
            {"role": "user", "content": "Reply with exactly: MEMORY_OK"}
        ])
        assert result  # just check it returns something

    @pytest.mark.asyncio
    async def test_ollama_embed(self):
        adapter = make_adapter(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            embedding_base_url="http://localhost:11434",
            embedding_api_key=None,
        )
        vector = await adapter.embed("test embedding from ollama")
        assert isinstance(vector, list)
        assert len(vector) > 0
