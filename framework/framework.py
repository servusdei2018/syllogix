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
from dataclasses import dataclass

from .engines.deductive import DeductiveEngine
from .llm import LLMResponseCache, ProviderConfig, ProviderRegistry
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

logger = logging.getLogger(__name__)


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

        self.llm_provider = ProviderRegistry.create_provider(
            self.config.provider, provider_config
        )

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

                prop_step = ReasoningStep(
                    step_id=3,
                    question="Proposition Formation",
                    reasoning_type="Observation",
                    syllogism=Syllogism(
                        major_premise=Proposition(
                            quantifier=major_prop.quantifier,
                            subject=major_prop.subject,
                            predicate=major_prop.predicate,
                        ),
                        minor_premise=Proposition(
                            quantifier=minor_prop.quantifier,
                            subject=minor_prop.subject,
                            predicate=minor_prop.predicate,
                        ),
                        conclusion=None,
                    ),
                    is_valid=True,
                    summary=f"Formed {len(propositions.propositions)} propositions",
                    confidence=0.7,
                )
                chain.add_step(prop_step)

                # Stage 4: Deductive Reasoning
                deductive_result = await self._apply_deductive_reasoning(propositions)
                if deductive_result and prop_step.syllogism:
                    major = prop_step.syllogism.major_premise
                    minor = prop_step.syllogism.minor_premise

                    conclusion = None
                    if (
                        deductive_result.valid
                        and deductive_result.conclusion_quantifier
                    ):
                        conclusion = Proposition(
                            quantifier=deductive_result.conclusion_quantifier,
                            subject=deductive_result.conclusion_subject or "",
                            predicate=deductive_result.conclusion_predicate or "",
                        )

                    deductive_step = ReasoningStep(
                        step_id=4,
                        question="Deductive Reasoning",
                        reasoning_type="Deductive",
                        syllogism=Syllogism(
                            major_premise=major,
                            minor_premise=minor,
                            conclusion=conclusion,
                        ),
                        mood=deductive_result.mood,
                        is_valid=deductive_result.valid,
                        summary=deductive_result.explanation,
                        confidence=1.0 if deductive_result.valid else 0.3,
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

    async def _analyze_query(self, query: str) -> QueryAnalysis | None:
        """Analyze the query using structured output."""
        prompt = f"""Analyze this query for logical reasoning:

Query: "{query}"

Provide a structured analysis of the query including the main topic, reasoning type required, key terms, and expected answer format."""

        return await self.llm_provider.generate_structured_async(prompt, QueryAnalysis)

    async def _retrieve_evidence(
        self, query: str, analysis: QueryAnalysis | None
    ) -> EvidenceCollection | None:
        """Retrieve evidence using structured output."""
        topic = analysis.main_topic if analysis else query
        prompt = f"""Provide factual evidence that can be used as premises to answer:

Query: "{query}"
Topic: {topic}

List specific factual statements that could serve as premises in a logical argument. Rate each fact's relevance and confidence."""

        return await self.llm_provider.generate_structured_async(
            prompt, EvidenceCollection
        )

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

Convert the evidence into 2-4 logical propositions using standard quantifiers (All, No, Some, Some...not). Specify which propositions should serve as major and minor premises for a syllogism."""

        return await self.llm_provider.generate_structured_async(prompt, PropositionSet)

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

        return await self.llm_provider.generate_structured_async(
            prompt, DeductiveConclusion
        )

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

        return await self.llm_provider.generate_structured_async(
            prompt, FinalConclusion
        )

    def reason_sync(self, query: str) -> ReasoningChain:
        """Synchronous wrapper for reason()."""
        return asyncio.run(self.reason(query))
