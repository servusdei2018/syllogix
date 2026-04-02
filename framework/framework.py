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

"""Core framework for syllogistic reasoning with structured LLM outputs.

This module provides the main LogicFramework class that orchestrates the entire
reasoning process using Pydantic-structured LLM outputs via LangChain.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from .engines.deductive import DeductiveEngine
from .llm import ConfigValidator, LLMResponseCache, ProviderConfig, ProviderRegistry
from .models import (
    Proposition,
    RAGSource,
    ReasoningChain,
    ReasoningStep,
    Syllogism,
)
from .schemas import (
    DeductiveConclusion,
    EvidenceCollection,
    FinalConclusion,
    PropositionSet,
    QueryAnalysis,
)
from .schemas import (
    Proposition as SchemaProposition,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _parse_env_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_env_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _resolve_api_key_from_env(provider: str) -> str:
    generic = os.environ.get("SYLLOGIX_API_KEY", "")
    if generic:
        return generic
    env_names = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    key = env_names.get(provider.lower())
    if key:
        return os.environ.get(key, "")
    return ""


def _model_proposition_from_schema(prop: SchemaProposition) -> Proposition:
    """Map schema proposition to domain model with trimmed terms."""
    return Proposition(
        quantifier=prop.quantifier,
        subject=prop.subject.strip(),
        predicate=prop.predicate.strip(),
    )


@dataclass
class FrameworkConfig:
    """Configuration for LogicFramework."""

    # LLM Provider Configuration
    provider: str = "openai"
    model: str = "gpt-5.2"
    api_key: str = ""
    base_url: str | None = None
    timeout: int = 30
    max_retries: int = 3

    # Caching Configuration
    enable_caching: bool = True
    cache_ttl: int = 3600

    # Reasoning Configuration
    max_reasoning_steps: int = 10
    min_confidence_threshold: float = 0.5

    # Logging Configuration
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> FrameworkConfig:
        """Build configuration from environment variables.

        Variables (all optional unless noted):

        - ``SYLLOGIX_PROVIDER`` — LLM provider id (default ``openai``).
        - ``SYLLOGIX_MODEL`` — model name (default ``gpt-5.2``).
        - ``SYLLOGIX_API_KEY`` — API key for any provider (preferred generic).
        - ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``,
          ``OPENROUTER_API_KEY`` — used when ``SYLLOGIX_API_KEY`` is unset,
          according to ``SYLLOGIX_PROVIDER``.
        - ``SYLLOGIX_BASE_URL`` — optional provider base URL.
        - ``SYLLOGIX_TIMEOUT``, ``SYLLOGIX_MAX_RETRIES`` — integers.
        - ``SYLLOGIX_ENABLE_CACHING`` — ``true``/``false``/``1``/``0`` (default true).
        - ``SYLLOGIX_CACHE_TTL`` — cache TTL seconds (default 3600).
        - ``SYLLOGIX_LOG_LEVEL`` — logging level name (default ``INFO``).

        ``max_reasoning_steps`` and ``min_confidence_threshold`` use class defaults.
        """
        provider = os.environ.get("SYLLOGIX_PROVIDER", "openai")
        api_key = _resolve_api_key_from_env(provider)
        base_url_raw = os.environ.get("SYLLOGIX_BASE_URL", "")
        return cls(
            provider=provider,
            model=os.environ.get("SYLLOGIX_MODEL", "gpt-5.2"),
            api_key=api_key,
            base_url=base_url_raw if base_url_raw else None,
            timeout=_parse_env_int(os.environ.get("SYLLOGIX_TIMEOUT"), 30),
            max_retries=_parse_env_int(os.environ.get("SYLLOGIX_MAX_RETRIES"), 3),
            enable_caching=_parse_env_bool(
                os.environ.get("SYLLOGIX_ENABLE_CACHING"), True
            ),
            cache_ttl=_parse_env_int(os.environ.get("SYLLOGIX_CACHE_TTL"), 3600),
            log_level=os.environ.get("SYLLOGIX_LOG_LEVEL", "INFO"),
        )


class LogicFramework:
    """Orchestrates the entire reasoning process with structured outputs."""

    def __init__(self, config: FrameworkConfig | None = None):
        """Initialize the LogicFramework.

        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config if config is not None else FrameworkConfig()

        self._setup_logging()
        self.deductive_engine = DeductiveEngine()
        self.llm_cache = LLMResponseCache(ttl=self.config.cache_ttl)

        self._init_llm_provider()
        logger.info(f"LogicFramework initialized with provider: {self.config.provider}")

    def _setup_logging(self):
        """Setup logging configuration."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def _init_llm_provider(self):
        """Initialize LLM provider."""
        provider_config = ProviderConfig(
            api_key=self.config.api_key,
            model=self.config.model,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

        errors = ConfigValidator.validate_provider_config(provider_config)
        if errors:
            raise ValueError("; ".join(errors))

        self.llm_provider = ProviderRegistry.create_provider(
            self.config.provider, provider_config
        )

    async def _structured_async(self, prompt: str, output_schema: type[T]) -> T | None:
        """Structured LLM call with optional response cache."""
        provider_name = self.llm_provider.get_provider_name()
        model = self.config.model
        schema_name = output_schema.__name__

        if self.config.enable_caching:
            cached = self.llm_cache.get(prompt, provider_name, model, schema_name)
            if cached is not None:
                return cached  # type: ignore[return-value]

        result = await self.llm_provider.generate_structured_async(
            prompt, output_schema
        )
        if self.config.enable_caching and result is not None:
            self.llm_cache.set(prompt, provider_name, model, schema_name, result)
        return result

    async def reason(self, query: str) -> ReasoningChain:
        """Execute the full reasoning chain for a query.

        Args:
            query: The query to reason about

        Returns:
            ReasoningChain containing the complete reasoning process
        """
        logger.info(f"Starting reasoning for query: {query}")
        chain = ReasoningChain(main_query=query)

        try:
            # Stage 1: Query Analysis
            analysis = await self._analyze_query(query)
            if analysis:
                analysis_step = ReasoningStep(
                    step_id=1,
                    question="Query Analysis",
                    reasoning_type="Observation",
                    rag_sources=[],
                    is_valid=True,
                    summary=f"Topic: {analysis.main_topic}. Type: {analysis.reasoning_type}",
                    confidence=0.9,
                )
                chain.add_step(analysis_step)

            # Stage 2: Evidence Retrieval
            evidence = await self._retrieve_evidence(query, analysis)
            if evidence:
                evidence_step = ReasoningStep(
                    step_id=2,
                    question="Evidence Retrieval",
                    reasoning_type="Observation",
                    rag_sources=[
                        RAGSource(source_id=f"ev_{i}", text=item.fact)
                        for i, item in enumerate(evidence.evidence_items)
                    ],
                    is_valid=True,
                    summary=evidence.summary,
                    confidence=0.8,
                )
                chain.add_step(evidence_step)

            # Stage 3: Proposition Formation
            propositions = await self._form_propositions(query, evidence)
            if propositions and len(propositions.propositions) >= 2:
                major_prop = propositions.propositions[propositions.major_premise_index]
                minor_prop = propositions.propositions[propositions.minor_premise_index]
                major_m = _model_proposition_from_schema(major_prop)
                minor_m = _model_proposition_from_schema(minor_prop)

                prop_step = ReasoningStep(
                    step_id=3,
                    question="Proposition Formation",
                    reasoning_type="Observation",
                    syllogism=Syllogism(
                        major_premise=major_m,
                        minor_premise=minor_m,
                        conclusion=None,
                    ),
                    is_valid=True,
                    summary=f"Formed {len(propositions.propositions)} propositions",
                    confidence=0.7,
                )
                chain.add_step(prop_step)

                # Stage 4: symbolic engine is authoritative; LLM supplies explanation only
                engine_input = ReasoningStep(
                    step_id=0,
                    question="Deductive Reasoning",
                    reasoning_type="Deductive",
                    syllogism=Syllogism(
                        major_premise=major_m,
                        minor_premise=minor_m,
                        conclusion=None,
                    ),
                    summary="",
                )
                validated = self.deductive_engine.validate(engine_input)
                llm_deductive = await self._apply_deductive_reasoning(propositions)
                deductive_step = self._build_deductive_reasoning_step(
                    validated, llm_deductive
                )
                chain.add_step(deductive_step)

            # Stage 5: Final Conclusion
            final = await self._generate_conclusion(query, chain)
            if final:
                final_step = ReasoningStep(
                    step_id=len(chain.steps) + 1,
                    question="Final Conclusion",
                    reasoning_type="Observation",
                    is_valid=final.is_valid,
                    summary=final.conclusion_text,
                    confidence=final.confidence,
                )
                chain.add_step(final_step)
                chain.final_conclusion_summary = final.conclusion_text

            logger.info(f"Reasoning completed successfully for query: {query}")

        except Exception as e:
            logger.error(f"Reasoning failed for query '{query}': {e}")
            error_step = ReasoningStep(
                step_id=len(chain.steps) + 1,
                question="Error Recovery",
                reasoning_type="Observation",
                is_valid=False,
                summary=f"Reasoning failed: {str(e)}",
                confidence=0.0,
            )
            chain.add_step(error_step)
            chain.final_conclusion_summary = f"Failed to reach conclusion: {str(e)}"

        return chain

    def _build_deductive_reasoning_step(
        self,
        validated: ReasoningStep,
        llm: DeductiveConclusion | None,
    ) -> ReasoningStep:
        """Merge symbolic engine result with optional LLM narrative (engine is authoritative)."""
        summary_parts: list[str] = []
        base = validated.summary.strip()
        if base:
            summary_parts.append(base)
        if llm and llm.explanation.strip():
            summary_parts.append(llm.explanation.strip())
        if llm is not None and llm.valid != validated.is_valid:
            summary_parts.append(
                "LLM structured assessment differed from symbolic validation."
            )
        merged = "\n\n".join(summary_parts)

        return ReasoningStep(
            step_id=4,
            question="Deductive Reasoning",
            reasoning_type="Deductive",
            syllogism=validated.syllogism,
            mood=validated.mood,
            is_valid=validated.is_valid,
            summary=merged,
            confidence=1.0 if validated.is_valid else 0.0,
        )

    async def _analyze_query(self, query: str) -> QueryAnalysis | None:
        """Analyze the query using structured output."""
        prompt = f"""Analyze this query for logical reasoning:

Query: "{query}"

Provide a structured analysis of the query including the main topic, reasoning type required, key terms, and expected answer format."""

        return await self._structured_async(prompt, QueryAnalysis)

    async def _retrieve_evidence(
        self, query: str, analysis: QueryAnalysis | None
    ) -> EvidenceCollection | None:
        """Retrieve evidence using structured output."""
        topic = analysis.main_topic if analysis else query
        prompt = f"""Provide factual evidence that can be used as premises to answer:

Query: "{query}"
Topic: {topic}

List specific factual statements that could serve as premises in a logical argument. Rate each fact's relevance and confidence."""

        return await self._structured_async(prompt, EvidenceCollection)

    async def _form_propositions(
        self, query: str, evidence: EvidenceCollection | None
    ) -> PropositionSet | None:
        """Form logical propositions from evidence."""
        if not evidence or not evidence.evidence_items:
            return None

        evidence_text = "\n".join(
            [f"- {item.fact}" for item in evidence.evidence_items]
        )

        prompt = f"""Form logical propositions from this evidence for answering:

Query: "{query}"

Evidence:
{evidence_text}

Convert the evidence into 2-4 logical propositions. Return a JSON object with exactly these fields:
- "propositions": array of objects, each with:
  - "quantifier": one of "All", "No", "Some", "Some...not", "Statistical"
  - "subject": the subject term (string)
  - "predicate": the predicate term (string)
  - "source_evidence": the evidence sentence this came from (string)
- "major_premise_index": integer index (0-based) of the universal/major premise
- "minor_premise_index": integer index (0-based) of the particular/minor premise

The major premise should be the universal statement (e.g., "All X are Y").
The minor premise should identify the subject as a member of the major premise category."""

        return await self._structured_async(prompt, PropositionSet)

    async def _apply_deductive_reasoning(
        self, propositions: PropositionSet
    ) -> DeductiveConclusion | None:
        """Apply deductive reasoning to the propositions."""
        if len(propositions.propositions) < 2:
            return None

        major = propositions.propositions[propositions.major_premise_index]
        minor = propositions.propositions[propositions.minor_premise_index]

        prompt = f"""Apply deductive reasoning to these premises:

Major Premise: {major.quantifier} {major.subject} are {major.predicate}
Minor Premise: {minor.quantifier} {minor.subject} are {minor.predicate}

Determine if these premises form a valid syllogism. If valid, identify the syllogistic mood (e.g., Barbara, Celarent) and state the conclusion. If invalid, explain why."""

        return await self._structured_async(prompt, DeductiveConclusion)

    async def _generate_conclusion(
        self, query: str, chain: ReasoningChain
    ) -> FinalConclusion | None:
        """Generate the final conclusion."""
        steps_summary = "\n".join(
            [f"Step {step.step_id}: {step.summary}" for step in chain.steps]
        )

        prompt = f"""Based on the following reasoning steps, provide a final conclusion for the query.

Query: {query}

Reasoning Steps:
{steps_summary}

State the final conclusion, whether it's logically valid, your confidence level, and a brief summary of the reasoning."""

        return await self._structured_async(prompt, FinalConclusion)

    def reason_sync(self, query: str) -> ReasoningChain:
        """Synchronous wrapper for reason()."""
        return asyncio.run(self.reason(query))
