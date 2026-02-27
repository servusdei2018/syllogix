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

"""Pydantic models for structured LLM outputs.

This module defines the structured output schemas used throughout the reasoning
pipeline, enabling type-safe LLM interactions via LangChain's structured output.
"""

from typing import List

from pydantic import BaseModel, Field

from .models import Quantifier


class QueryAnalysis(BaseModel):
    """Structured analysis of a user query."""

    main_topic: str = Field(description="The main subject or topic of the query")
    reasoning_type: str = Field(
        description="The type of reasoning required (deductive, inductive, abductive, etc.)"
    )
    key_terms: List[str] = Field(
        description="Key terms and concepts mentioned in the query"
    )
    expected_answer_format: str = Field(
        description="The expected format or structure of the answer"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Any implicit assumptions or context needed to answer the query",
    )


class EvidenceItem(BaseModel):
    """A single piece of evidence retrieved for reasoning."""

    fact: str = Field(description="A factual statement that can serve as a premise")
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="How relevant this fact is to the query (0-1)"
    )
    confidence: str = Field(
        pattern="^(high|medium|low|uncertain)$",
        description="Confidence level in this fact",
    )


class EvidenceCollection(BaseModel):
    """Collection of evidence retrieved for a query."""

    summary: str = Field(description="A concise summary of the key evidence points")
    evidence_items: List[EvidenceItem] = Field(
        description="List of individual evidence items"
    )


class Proposition(BaseModel):
    """A logical proposition extracted from evidence."""

    quantifier: Quantifier
    subject: str = Field(description="The subject term of the proposition")
    predicate: str = Field(description="The predicate term of the proposition")
    source_evidence: str = Field(
        description="The evidence this proposition was derived from"
    )


class PropositionSet(BaseModel):
    """A set of logical propositions for syllogistic reasoning."""

    propositions: List[Proposition] = Field(
        description="List of extracted propositions", min_length=1, max_length=4
    )
    major_premise_index: int = Field(
        ge=0, description="Index of the proposition to use as major premise"
    )
    minor_premise_index: int = Field(
        ge=0, description="Index of the proposition to use as minor premise"
    )


class DeductiveConclusion(BaseModel):
    """Result of deductive reasoning."""

    valid: bool = Field(description="Whether the syllogism is valid")
    mood: str | None = Field(
        default=None,
        description="The syllogistic mood if valid (e.g., Barbara, Celarent)",
    )
    conclusion_quantifier: Quantifier | None = Field(
        default=None, description="The quantifier of the conclusion"
    )
    conclusion_subject: str | None = Field(
        default=None, description="The subject of the conclusion"
    )
    conclusion_predicate: str | None = Field(
        default=None, description="The predicate of the conclusion"
    )
    explanation: str = Field(
        description="Explanation of the reasoning or why it's invalid"
    )


class FinalConclusion(BaseModel):
    """The final conclusion of the reasoning process."""

    conclusion_text: str = Field(description="The final conclusion statement")
    is_valid: bool = Field(description="Whether the conclusion is logically valid")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the conclusion (0-1)"
    )
    reasoning_summary: str = Field(
        description="Summary of how the conclusion was reached"
    )
