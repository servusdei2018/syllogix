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

from framework.config import load_framework_config_from_env
from framework.framework import FrameworkConfig, LogicFramework
from framework.llm import LLMProvider, ProviderConfig
from framework.schemas import (
    DeductiveConclusion,
    EvidenceCollection,
    EvidenceItem,
    FinalConclusion,
    Proposition,
    PropositionSet,
    QueryAnalysis,
)


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

    def test_raises_when_api_key_missing(self):
        """ConfigValidator requires an API key at startup."""
        with pytest.raises(ValueError, match="API key"):
            LogicFramework(FrameworkConfig())


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

    @staticmethod
    def _empty_evidence_framework(enable_caching: bool) -> LogicFramework:
        config = FrameworkConfig(api_key="test-key", enable_caching=enable_caching)
        framework = LogicFramework(config)
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
    async def test_caching_second_run_reuses_llm_cache(self) -> None:
        """Identical prompts should not call the provider again when caching is on."""
        fw = self._empty_evidence_framework(enable_caching=True)
        mock = fw.llm_provider
        assert isinstance(mock, MockLLMProvider)
        await fw.reason("What is logic?")
        first_count = mock.call_count
        await fw.reason("What is logic?")
        assert mock.call_count == first_count

    @pytest.mark.asyncio
    async def test_caching_disabled_calls_provider_each_time(self) -> None:
        """With caching off, each full reason() issues fresh LLM calls."""
        fw = self._empty_evidence_framework(enable_caching=False)
        mock = fw.llm_provider
        assert isinstance(mock, MockLLMProvider)
        await fw.reason("What is logic?")
        first_count = mock.call_count
        await fw.reason("What is logic?")
        assert mock.call_count == first_count * 2


_ENV_KEYS_TO_CLEAR = (
    "SYLLOGIX_PROVIDER",
    "SYLLOGIX_MODEL",
    "SYLLOGIX_API_KEY",
    "SYLLOGIX_BASE_URL",
    "SYLLOGIX_TIMEOUT",
    "SYLLOGIX_MAX_RETRIES",
    "SYLLOGIX_ENABLE_CACHING",
    "SYLLOGIX_CACHE_TTL",
    "SYLLOGIX_LOG_LEVEL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)


def _clear_syllogix_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS_TO_CLEAR:
        monkeypatch.delenv(key, raising=False)


class TestFrameworkConfigFromEnv:
    """Tests for FrameworkConfig.from_env and load_framework_config_from_env."""

    def test_from_env_syllogix_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_syllogix_env(monkeypatch)
        monkeypatch.setenv("SYLLOGIX_API_KEY", "key-from-syllogix")
        cfg = FrameworkConfig.from_env()
        assert cfg.api_key == "key-from-syllogix"

    def test_from_env_openai_api_key_when_generic_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_syllogix_env(monkeypatch)
        monkeypatch.setenv("SYLLOGIX_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "key-from-openai")
        cfg = FrameworkConfig.from_env()
        assert cfg.api_key == "key-from-openai"
        assert cfg.provider == "openai"

    def test_from_env_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_syllogix_env(monkeypatch)
        monkeypatch.setenv("SYLLOGIX_API_KEY", "x")
        monkeypatch.setenv("SYLLOGIX_MODEL", "custom-model-id")
        cfg = FrameworkConfig.from_env()
        assert cfg.model == "custom-model-id"

    def test_load_framework_config_from_env_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_syllogix_env(monkeypatch)
        monkeypatch.setenv("SYLLOGIX_API_KEY", "alias-test")
        cfg = load_framework_config_from_env()
        assert cfg.api_key == "alias-test"


def _evidence_one_fact() -> EvidenceCollection:
    return EvidenceCollection(
        summary="Stub evidence",
        evidence_items=[
            EvidenceItem(
                fact="Premise one.",
                relevance_score=1.0,
                confidence="high",
            ),
        ],
    )


def _barbara_proposition_set() -> PropositionSet:
    return PropositionSet(
        propositions=[
            Proposition(
                quantifier="All",
                subject="M",
                predicate="P",
                source_evidence="major",
            ),
            Proposition(
                quantifier="All",
                subject="S",
                predicate="M",
                source_evidence="minor",
            ),
        ],
        major_premise_index=0,
        minor_premise_index=1,
    )


class TestDeductiveReconciliation:
    """Symbolic engine is authoritative over LLM structured DeductiveConclusion."""

    @pytest.mark.asyncio
    async def test_engine_wins_when_llm_structured_output_conflicts(self) -> None:
        """Barbara premises: engine valid Barbara; LLM says invalid — chain follows engine."""
        config = FrameworkConfig(api_key="test-key")
        framework = LogicFramework(config)
        framework.llm_provider = MockLLMProvider(
            {
                "QueryAnalysis": QueryAnalysis(
                    main_topic="Logic",
                    reasoning_type="Deductive",
                    key_terms=["syllogism"],
                    expected_answer_format="Text",
                    assumptions=[],
                ),
                "EvidenceCollection": _evidence_one_fact(),
                "PropositionSet": _barbara_proposition_set(),
                "DeductiveConclusion": DeductiveConclusion(
                    valid=False,
                    mood="Celarent",
                    conclusion_quantifier="Some",
                    conclusion_subject="X",
                    conclusion_predicate="Y",
                    explanation="Structured LLM claims invalid.",
                ),
                "FinalConclusion": FinalConclusion(
                    conclusion_text="Done",
                    is_valid=True,
                    confidence=0.5,
                    reasoning_summary="R",
                ),
            }
        )

        chain = await framework.reason("Barbara test")
        deductive = next(s for s in chain.steps if s.step_id == 4)

        assert deductive.is_valid is True
        assert deductive.mood == "Barbara"
        assert deductive.syllogism is not None
        assert deductive.syllogism.conclusion is not None
        assert deductive.syllogism.conclusion.quantifier == "All"
        assert deductive.syllogism.conclusion.subject == "S"
        assert deductive.syllogism.conclusion.predicate == "P"
        assert deductive.confidence == 1.0
        assert "Structured LLM claims invalid." in deductive.summary
        assert (
            "LLM structured assessment differed from symbolic validation."
            in deductive.summary
        )

    @pytest.mark.asyncio
    async def test_engine_invalid_premises_llm_valid_disagreement_note(self) -> None:
        """No classical mood: engine invalid; LLM valid=True — chain follows engine."""
        config = FrameworkConfig(api_key="test-key")
        framework = LogicFramework(config)
        framework.llm_provider = MockLLMProvider(
            {
                "QueryAnalysis": QueryAnalysis(
                    main_topic="Logic",
                    reasoning_type="Deductive",
                    key_terms=["syllogism"],
                    expected_answer_format="Text",
                    assumptions=[],
                ),
                "EvidenceCollection": _evidence_one_fact(),
                "PropositionSet": PropositionSet(
                    propositions=[
                        Proposition(
                            quantifier="All",
                            subject="A",
                            predicate="B",
                            source_evidence="a",
                        ),
                        Proposition(
                            quantifier="All",
                            subject="C",
                            predicate="D",
                            source_evidence="b",
                        ),
                    ],
                    major_premise_index=0,
                    minor_premise_index=1,
                ),
                "DeductiveConclusion": DeductiveConclusion(
                    valid=True,
                    mood="Barbara",
                    conclusion_quantifier="All",
                    conclusion_subject="A",
                    conclusion_predicate="D",
                    explanation="LLM thinks this is Barbara.",
                ),
                "FinalConclusion": FinalConclusion(
                    conclusion_text="Done",
                    is_valid=False,
                    confidence=0.4,
                    reasoning_summary="R",
                ),
            }
        )

        chain = await framework.reason("Invalid mix")
        deductive = next(s for s in chain.steps if s.step_id == 4)

        assert deductive.is_valid is False
        assert deductive.syllogism is not None
        assert deductive.syllogism.conclusion is None
        assert deductive.confidence == 0.0
        assert "[EngineError:" in deductive.summary
        assert "LLM thinks this is Barbara." in deductive.summary
        assert (
            "LLM structured assessment differed from symbolic validation."
            in deductive.summary
        )
