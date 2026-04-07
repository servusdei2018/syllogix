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

import json
import re
from typing import Any, List, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .models import Quantifier


def _as_str_key_dict(data: dict[Any, Any]) -> dict[str, Any]:
    """Shallow copy of arbitrary dict payloads from the LLM into ``dict[str, Any]``."""
    return {str(k): v for k, v in data.items()}


def _parse_quantifier_form(qf: str) -> tuple[str, str, str] | None:
    """Parse 'All H are M' / 'Some X are not Y' / 'S is H' → (quantifier, subject, predicate)."""
    qf = qf.strip()
    # Some...not  e.g. "Some S are not P"
    m = re.match(r"^(All|No|Some)\s+(.+?)\s+are\s+not\s+(.+)$", qf, re.IGNORECASE)
    if m:
        return "Some...not", m.group(2).strip(), m.group(3).strip()
    # All / No / Some  e.g. "All H are M"
    m = re.match(r"^(All|No|Some)\s+(.+?)\s+are\s+(.+)$", qf, re.IGNORECASE)
    if m:
        q = m.group(1).capitalize()
        return q, m.group(2).strip(), m.group(3).strip()
    # Singular  e.g. "S is H"
    m = re.match(r"^(.+?)\s+(?:is|are)\s+(.+)$", qf, re.IGNORECASE)
    if m:
        return "All", m.group(1).strip(), m.group(2).strip()
    return None


def _stringify_key_term_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        d = cast(dict[str, Any], item)
        term = str(d.get("term", "")).strip()
        role = str(d.get("role", "")).strip()
        notes = str(d.get("notes", "")).strip()
        parts: list[str] = []
        if term:
            parts.append(term)
        if role:
            parts.append(f"({role})")
        if notes:
            parts.append(f": {notes}")
        return " ".join(parts).strip() if parts else json.dumps(d, ensure_ascii=False)
    return str(item)


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

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_variants(cls, data: Any) -> Any:
        """Map common alternate JSON shapes from chat models into this schema."""
        if not isinstance(data, dict):
            return data
        out = _as_str_key_dict(cast(dict[Any, Any], data))
        if out.get("reasoning_type") is None and "reasoning_type_required" in out:
            rtr: Any = out.pop("reasoning_type_required")
            out["reasoning_type"] = (
                "; ".join(str(x) for x in cast(list[Any], rtr))
                if isinstance(rtr, list)
                else str(rtr)
            )
        kt: Any = out.get("key_terms")
        if isinstance(kt, list) and kt:
            out["key_terms"] = [
                _stringify_key_term_item(x) for x in cast(list[Any], kt)
            ]
        eaf: Any = out.get("expected_answer_format")
        if isinstance(eaf, (dict, list)):
            out["expected_answer_format"] = json.dumps(eaf, ensure_ascii=False)
        return out


class EvidenceItem(BaseModel):
    """A single piece of evidence retrieved for reasoning."""

    model_config = ConfigDict(populate_by_name=True)

    fact: str = Field(
        validation_alias=AliasChoices("fact", "statement"),
        description="A factual statement that can serve as a premise",
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("relevance_score", "relevance"),
        description="How relevant this fact is to the query (0-1)",
    )
    confidence: str = Field(
        pattern="^(high|medium|low|uncertain)$",
        description="Confidence level in this fact",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> str:
        if isinstance(v, (int, float)):
            x = float(v)
            if x >= 0.75:
                return "high"
            if x >= 0.45:
                return "medium"
            if x >= 0.2:
                return "low"
            return "uncertain"
        s = str(v).strip().lower()
        if s in ("high", "medium", "low", "uncertain"):
            return s
        return "medium"


class EvidenceCollection(BaseModel):
    """Collection of evidence retrieved for a query."""

    summary: str = Field(description="A concise summary of the key evidence points")
    evidence_items: List[EvidenceItem] = Field(
        description="List of individual evidence items"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_variants(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = _as_str_key_dict(cast(dict[Any, Any], data))
        if "evidence_items" not in out and "premises" in out:
            out["evidence_items"] = out.pop("premises")
        if not out.get("summary"):
            topic_q = str(out.get("topic") or out.get("query") or "").strip()
            out["summary"] = topic_q or "Summary of retrieved evidence."
        return out


class Proposition(BaseModel):
    """A logical proposition extracted from evidence."""

    quantifier: Quantifier
    subject: str = Field(description="The subject term of the proposition")
    predicate: str = Field(description="The predicate term of the proposition")
    source_evidence: str = Field(
        description="The evidence this proposition was derived from"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_variants(cls, data: Any) -> Any:
        """Handle alternate LLM shapes (quantifier_form, text, id, role)."""
        if not isinstance(data, dict):
            return data
        out = _as_str_key_dict(cast(dict[Any, Any], data))

        # source_evidence can come from 'text' or 'id'
        if not out.get("source_evidence"):
            out["source_evidence"] = str(out.get("text") or out.get("id") or "").strip()

        # Parse missing quantifier/subject/predicate from 'quantifier_form'
        qf = str(out.get("quantifier_form") or "").strip()
        needs_parse = (
            not out.get("quantifier")
            or not out.get("subject")
            or not out.get("predicate")
        )
        if qf and needs_parse:
            parsed = _parse_quantifier_form(qf)
            if parsed:
                q, s, p = parsed
                if not out.get("quantifier"):
                    out["quantifier"] = q
                if not out.get("subject"):
                    out["subject"] = s
                if not out.get("predicate"):
                    out["predicate"] = p

        # Last-resort: derive from 'text' field
        text = str(out.get("text") or "").strip()
        if text and needs_parse:
            parsed = _parse_quantifier_form(text)
            if parsed:
                q, s, p = parsed
                if not out.get("quantifier"):
                    out["quantifier"] = q
                if not out.get("subject"):
                    out["subject"] = s
                if not out.get("predicate"):
                    out["predicate"] = p

        # Handle Extended Figures: Treat singular propositions as Universal
        q = out.get("quantifier")
        s = out.get("subject")
        if q == "Some" and s:
            try:
                from .nlp import is_singular_term

                if is_singular_term(s):
                    out["quantifier"] = "All"
            except ImportError:
                pass

        return out


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

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_variants(cls, data: Any) -> Any:
        """Derive major/minor premise indices from role fields or syllogisms array."""
        if not isinstance(data, dict):
            return data
        out = _as_str_key_dict(cast(dict[Any, Any], data))
        props = out.get("propositions") or []

        if not isinstance(props, list):
            return out

        def _find_by_role(role: str) -> int | None:
            for i, p in enumerate(props):
                if isinstance(p, dict) and str(p.get("role", "")).strip() == role:
                    return i
            return None

        def _find_by_id(pid: Any) -> int | None:
            for i, p in enumerate(props):
                if isinstance(p, dict) and p.get("id") == pid:
                    return i
            return None

        # Prefer role-based lookup for major premise
        if "major_premise_index" not in out:
            idx = _find_by_role("major_premise") or _find_by_role("universal_premise")
            if idx is not None:
                out["major_premise_index"] = idx

        # Prefer role-based lookup for minor premise
        if "minor_premise_index" not in out:
            idx = _find_by_role("minor_premise") or _find_by_role("particular_premise")
            if idx is not None:
                out["minor_premise_index"] = idx

        # Fall back to first syllogism's premise IDs
        syllogisms = out.get("syllogisms") or []
        if isinstance(syllogisms, list) and syllogisms:
            first = syllogisms[0]
            if isinstance(first, dict):
                if "major_premise_index" not in out:
                    idx = _find_by_id(first.get("major_premise_id"))
                    if idx is not None:
                        out["major_premise_index"] = idx
                if "minor_premise_index" not in out:
                    idx = _find_by_id(first.get("minor_premise_id"))
                    if idx is not None:
                        out["minor_premise_index"] = idx

        # Ultimate fallback: 0 / 1 when we have at least two propositions
        if "major_premise_index" not in out and len(props) >= 2:
            out["major_premise_index"] = 0
        if "minor_premise_index" not in out and len(props) >= 2:
            out["minor_premise_index"] = 1

        return out

    @model_validator(mode="after")
    def premise_indices_in_range_and_distinct(self) -> "PropositionSet":
        n = len(self.propositions)
        if self.major_premise_index >= n:
            raise ValueError("major_premise_index must be less than len(propositions)")
        if self.minor_premise_index >= n:
            raise ValueError("minor_premise_index must be less than len(propositions)")
        if self.major_premise_index == self.minor_premise_index:
            raise ValueError("major_premise_index and minor_premise_index must differ")
        return self


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

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_variants(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = _as_str_key_dict(cast(dict[Any, Any], data))
        if "conclusion_text" not in out:
            if "final_conclusion" in out:
                out["conclusion_text"] = out.pop("final_conclusion")
            elif "conclusion" in out:
                out["conclusion_text"] = out.pop("conclusion")
        if "is_valid" not in out and "logically_valid" in out:
            out["is_valid"] = out.pop("logically_valid")
        if "reasoning_summary" not in out:
            for alt in ("summary", "reasoning", "explanation", "rationale"):
                if alt in out:
                    out["reasoning_summary"] = out.pop(alt)
                    break
        return out
