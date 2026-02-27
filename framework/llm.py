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

"""LangChain-based LLM provider system with structured outputs.

This module provides a unified interface for multiple LLM providers using LangChain,
enabling type-safe structured outputs via Pydantic models.
"""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    api_key: str
    model: str
    base_url: str | None = None
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.5
    temperature: float = 0.7
    max_tokens: int = 4000


class LLMProvider(ABC):
    """Abstract base class for LLM providers using LangChain."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client: Any = self._create_client()

    @abstractmethod
    def _create_client(self) -> Any:
        """Create the LangChain chat model client."""
        pass

    def generate_structured(
        self, prompt: str, output_schema: type[T], **kwargs: Any
    ) -> T | None:
        """Generate structured output using the LLM.

        Args:
            prompt: The prompt to send to the LLM
            output_schema: Pydantic model class defining the expected output
            **kwargs: Additional parameters for the LLM

        Returns:
            Instance of output_schema or None if generation failed
        """
        try:
            # Bind the structured output schema
            structured_llm = self._client.with_structured_output(
                output_schema, method="json_mode"
            )

            # Generate the response
            result: T = structured_llm.invoke(prompt, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            return None

    async def generate_structured_async(
        self, prompt: str, output_schema: type[T], **kwargs: Any
    ) -> T | None:
        """Generate structured output asynchronously.

        Args:
            prompt: The prompt to send to the LLM
            output_schema: Pydantic model class defining the expected output
            **kwargs: Additional parameters for the LLM

        Returns:
            Instance of output_schema or None if generation failed
        """
        try:
            # Bind the structured output schema
            structured_llm = self._client.with_structured_output(
                output_schema, method="json_mode"
            )

            # Generate the response
            result: T = await structured_llm.ainvoke(prompt, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Async structured generation failed: {e}")
            return None

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider using LangChain."""

    def _create_client(self) -> ChatOpenAI:
        """Create OpenAI LangChain client."""
        return ChatOpenAI(
            model=self.config.model,
            api_key=SecretStr(self.config.api_key),
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_tokens,
        )

    def get_provider_name(self) -> str:
        return "openai"


class AnthropicProvider(LLMProvider):
    """Anthropic provider using LangChain."""

    def _create_client(self) -> ChatAnthropic:
        """Create Anthropic LangChain client."""
        return ChatAnthropic(
            model_name=self.config.model,
            api_key=SecretStr(self.config.api_key),
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def get_provider_name(self) -> str:
        return "anthropic"


class GoogleProvider(LLMProvider):
    """Google Gemini provider using LangChain."""

    def _create_client(self) -> ChatGoogleGenerativeAI:
        """Create Google LangChain client."""
        return ChatGoogleGenerativeAI(
            model=self.config.model,
            api_key=SecretStr(self.config.api_key),
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

    def get_provider_name(self) -> str:
        return "google"


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider using LangChain (OpenAI-compatible)."""

    def _create_client(self) -> ChatOpenAI:
        """Create OpenRouter LangChain client."""
        base_url = self.config.base_url or "https://openrouter.ai/api/v1"
        return ChatOpenAI(
            model=self.config.model,
            api_key=SecretStr(self.config.api_key),
            base_url=base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_tokens,
            extra_headers={
                "HTTP-Referer": "https://github.com/syllogix",
                "X-Title": "Syllogix",
            },
        )

    def get_provider_name(self) -> str:
        return "openrouter"


class ProviderRegistry:
    """Registry for LLM providers."""

    PROVIDERS: dict[str, type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "openrouter": OpenRouterProvider,
    }

    @classmethod
    def create_provider(cls, provider_type: str, config: ProviderConfig) -> LLMProvider:
        """Create a provider instance."""
        provider_class = cls.PROVIDERS.get(provider_type)
        if not provider_class:
            raise ValueError(
                f"Unsupported provider: {provider_type}. Supported: {list(cls.PROVIDERS.keys())}"
            )
        return provider_class(config)

    @classmethod
    def get_supported_models(cls, provider_type: str) -> list[str]:
        """Get supported models for a provider."""
        models = {
            "openai": [
                "gpt-5.2",
                "gpt-5-mini",
            ],
            "anthropic": [
                "claude-sonnet-4-6",
                "claude-haiku-4-5",
            ],
            "google": [
                "gemini-3-flash-preview",
                "gemini-3.1-pro-preview",
            ],
            "openrouter": [
                "moonshotai/kimi-k2.5",
                "z-ai/glm-5",
            ],
        }
        return models.get(provider_type, [])


class LLMResponseCache:
    """Simple in-memory cache for LLM responses."""

    def __init__(self, ttl: int = 3600, maxsize: int = 1000):
        self.ttl: int = ttl
        self.maxsize: int = maxsize
        self.cache: dict[str, Any] = {}
        self.timestamps: dict[str, float] = {}

    def _generate_key(
        self,
        prompt: str,
        provider: str,
        model: str,
        schema_name: str,
    ) -> str:
        """Generate cache key."""
        key_data = f"{prompt}|{provider}|{model}|{schema_name}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl

    def _cleanup(self) -> None:
        """Clean up expired entries."""
        expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]

    def get(
        self,
        prompt: str,
        provider: str,
        model: str,
        schema_name: str,
    ) -> Any | None:
        """Get cached response."""
        self._cleanup()

        if len(self.cache) >= self.maxsize:
            # Remove oldest entry
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]

        key = self._generate_key(prompt, provider, model, schema_name)

        if key in self.cache and not self._is_expired(key):
            logger.debug(f"Cache hit for key: {key[:8]}...")
            return self.cache[key]

        return None

    def set(
        self,
        prompt: str,
        provider: str,
        model: str,
        schema_name: str,
        response: Any,
    ) -> None:
        """Set cache response."""
        key = self._generate_key(prompt, provider, model, schema_name)
        self.cache[key] = response
        self.timestamps[key] = time.time()
        logger.debug(f"Cached response for key: {key[:8]}...")


class LLMHealthMonitor:
    """Health monitoring for LLM providers."""

    def __init__(self):
        self.health_status: dict[str, dict[str, Any]] = {}

    def check_health(self, provider: LLMProvider) -> dict[str, Any]:
        """Check provider health."""

        class HealthCheck(BaseModel):
            status: str

        provider_name = provider.get_provider_name()

        try:
            start = time.time()
            result = provider.generate_structured(
                "Return status 'healthy'", HealthCheck
            )
            latency = time.time() - start

            is_healthy = result is not None

            self.health_status[provider_name] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "latency": latency,
                "error": None if is_healthy else "Generation failed",
                "last_check": datetime.now().isoformat(),
            }
        except Exception as e:
            self.health_status[provider_name] = {
                "status": "unhealthy",
                "latency": None,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }

        return self.health_status[provider_name]

    def get_health_status(self, provider_name: str) -> dict[str, Any] | None:
        """Get health status for a provider."""
        return self.health_status.get(provider_name)


class ConfigValidator:
    """Configuration validation."""

    @staticmethod
    def validate_provider_config(config: ProviderConfig) -> list[str]:
        """Validate provider configuration."""
        errors: list[str] = []

        if not config.api_key:
            errors.append("API key is required")

        if config.timeout <= 0:
            errors.append("Timeout must be positive")

        if config.max_retries < 0:
            errors.append("Max retries must be non-negative")

        return errors
