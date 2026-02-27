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

"""Tests for LogicFramework with structured outputs."""

from typing import Any

import pytest

from framework.framework import FrameworkConfig, LogicFramework
from framework.llm import LLMProvider, ProviderConfig
from framework.schemas import EvidenceCollection, FinalConclusion, QueryAnalysis


class TestFrameworkConfig:
    """Test FrameworkConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FrameworkConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-5.2"
        assert config.api_key == ""
        assert config.enable_caching is True
        assert config.max_reasoning_steps == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FrameworkConfig(
            provider="anthropic",
            model="claude-opus-4-1",
            api_key="test-key",
            enable_caching=False,
            max_reasoning_steps=5,
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-opus-4-1"
        assert config.api_key == "test-key"
        assert config.enable_caching is False
        assert config.max_reasoning_steps == 5


class TestLogicFrameworkInitialization:
    """Test LogicFramework initialization."""

    def test_default_initialization(self):
        """Test initialization with default config."""
        config = FrameworkConfig(api_key="test-key")
        framework = LogicFramework(config)

        assert framework.config == config
        assert framework.deductive_engine is not None
        assert framework.llm_cache is not None


class MockLLMProvider(LLMProvider):
    """Mock provider for testing framework."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        # Initialize parent with mock config
        super().__init__(ProviderConfig(api_key="mock", model="mock"))
        self.responses: dict[str, Any] = responses or {}
        self.call_count: int = 0

    def _create_client(self) -> Any:
        return None

    async def generate_structured_async(
        self, prompt: str, output_schema: type[Any], **kwargs: Any
    ) -> Any | None:
        self.call_count += 1
        schema_name = output_schema.__name__
        if schema_name in self.responses:
            return self.responses[schema_name]
        return None

    def get_provider_name(self) -> str:
        return "mock"


class TestLogicFrameworkReasoning:
    """Test LogicFramework reasoning with structured outputs."""

    @pytest.fixture
    def mock_framework(self):
        """Create a framework with mocked provider."""
        config = FrameworkConfig(api_key="test-key")
        framework = LogicFramework(config)

        # Set up mock responses
        framework.llm_provider = MockLLMProvider(
            {
                "QueryAnalysis": QueryAnalysis(
                    main_topic="Logic",
                    reasoning_type="Deductive",
                    key_terms=["syllogism", "validity"],
                    expected_answer_format="Boolean",
                    assumptions=[],
                ),
                "EvidenceCollection": EvidenceCollection(
                    summary="Evidence about logic", evidence_items=[]
                ),
                "FinalConclusion": FinalConclusion(
                    conclusion_text="Test conclusion",
                    is_valid=True,
                    confidence=0.9,
                    reasoning_summary="Test reasoning",
                ),
            }
        )

        return framework

    @pytest.mark.asyncio
    async def test_full_reasoning_chain(self, mock_framework: LogicFramework) -> None:
        """Test full reasoning chain execution."""
        from framework.models import ReasoningChain

        result = await mock_framework.reason("What is logic?")

        assert isinstance(result, ReasoningChain)
        assert result.main_query == "What is logic?"
        assert len(result.steps) > 0
        assert result.final_conclusion_summary is not None

    def test_reason_sync(self, mock_framework: LogicFramework) -> None:
        """Test synchronous reasoning wrapper."""
        from framework.models import ReasoningChain

        result = mock_framework.reason_sync("What is logic?")

        assert isinstance(result, ReasoningChain)
        assert result.main_query == "What is logic?"
