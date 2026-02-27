# Syllogix
# Copyright (C) 2026  Nathanael Bracy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for the structured LLM provider system."""

from typing import Any

import pytest
from pydantic import BaseModel

from framework.llm import (
    ConfigValidator,
    LLMProvider,
    LLMResponseCache,
    ProviderConfig,
    ProviderRegistry,
)


class TestOutput(BaseModel):
    """Test output schema."""

    message: str
    count: int


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def _create_client(self) -> Any:
        """Return a mock client."""
        return MockChatModel()

    def get_provider_name(self) -> str:
        return "mock"


class MockChatModel:
    """Mock LangChain chat model."""

    def with_structured_output(
        self, schema: type[Any], method: str = "json_mode"
    ) -> "MockStructuredLLM":
        return MockStructuredLLM(schema)


class MockStructuredLLM:
    """Mock structured output LLM."""

    def __init__(self, schema: type[Any]):
        self.schema = schema

    def invoke(self, _prompt: str, **_kwargs: Any) -> Any:
        """Return a mock result."""
        if self.schema == TestOutput:
            return TestOutput(message="Test response", count=42)
        return None

    async def ainvoke(self, prompt: str, **kwargs: Any) -> Any:
        """Return a mock result asynchronously."""
        return self.invoke(prompt, **kwargs)


class TestProviderConfig:
    """Test ProviderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProviderConfig(api_key="test-key", model="test-model")
        assert config.api_key == "test-key"
        assert config.model == "test-model"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.temperature == 0.7
        assert config.max_tokens == 4000
        assert config.base_url is None


class TestProviderRegistry:
    """Test ProviderRegistry."""

    def test_supported_providers(self):
        """Test that all expected providers are registered."""
        expected = ["openai", "anthropic", "google", "openrouter"]
        for provider in expected:
            assert provider in ProviderRegistry.PROVIDERS

    def test_unsupported_provider(self) -> None:
        """Test that unsupported providers raise ValueError."""
        config = ProviderConfig(api_key="test", model="test")
        with pytest.raises(ValueError):
            _ = ProviderRegistry.create_provider("unsupported", config)

    def test_get_supported_models(self):
        """Test getting supported models for providers."""
        models = ProviderRegistry.get_supported_models("openai")
        assert len(models) > 0
        assert "gpt-5.2" in models


class TestLLMResponseCache:
    """Test LLMResponseCache."""

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LLMResponseCache()
        result = cache.get("prompt", "provider", "model", "TestOutput")
        assert result is None

    def test_cache_hit(self):
        """Test cache hit returns cached response."""
        cache = LLMResponseCache()
        response = TestOutput(message="test", count=1)

        cache.set("prompt", "provider", "model", "TestOutput", response)
        result = cache.get("prompt", "provider", "model", "TestOutput")

        assert result == response

    def test_cache_different_schemas(self):
        """Test cache distinguishes between different schemas."""
        cache = LLMResponseCache()
        response1 = TestOutput(message="test1", count=1)
        response2 = TestOutput(message="test2", count=2)

        cache.set("prompt", "provider", "model", "Schema1", response1)
        cache.set("prompt", "provider", "model", "Schema2", response2)

        assert cache.get("prompt", "provider", "model", "Schema1") == response1
        assert cache.get("prompt", "provider", "model", "Schema2") == response2


class TestConfigValidator:
    """Test ConfigValidator."""

    def test_valid_config(self):
        """Test validation of valid config."""
        config = ProviderConfig(api_key="valid-key", model="model")
        errors = ConfigValidator.validate_provider_config(config)
        assert len(errors) == 0

    def test_missing_api_key(self):
        """Test validation catches missing API key."""
        config = ProviderConfig(api_key="", model="model")
        errors = ConfigValidator.validate_provider_config(config)
        assert "API key is required" in errors

    def test_invalid_timeout(self):
        """Test validation catches invalid timeout."""
        config = ProviderConfig(api_key="key", model="model", timeout=-1)
        errors = ConfigValidator.validate_provider_config(config)
        assert "Timeout must be positive" in errors

    def test_invalid_retries(self):
        """Test validation catches invalid retries."""
        config = ProviderConfig(api_key="key", model="model", max_retries=-1)
        errors = ConfigValidator.validate_provider_config(config)
        assert "Max retries must be non-negative" in errors


class TestMockProviderStructuredOutput:
    """Test structured output with mock provider."""

    @pytest.fixture
    def mock_provider(self) -> MockProvider:
        """Create a mock provider."""
        return MockProvider(ProviderConfig(api_key="test", model="test"))

    def test_generate_structured(self, mock_provider: MockProvider) -> None:
        """Test synchronous structured generation."""
        result = mock_provider.generate_structured("Test prompt", TestOutput)
        assert result is not None
        assert result.message == "Test response"
        assert result.count == 42

    @pytest.mark.asyncio
    async def test_generate_structured_async(self, mock_provider: MockProvider) -> None:
        """Test asynchronous structured generation."""
        result = await mock_provider.generate_structured_async(
            "Test prompt", TestOutput
        )
        assert result is not None
        assert result.message == "Test response"
        assert result.count == 42
