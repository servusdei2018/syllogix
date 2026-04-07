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

import pytest
from pydantic import ValidationError

from framework.schemas import (
    EvidenceCollection,
    FinalConclusion,
    Proposition,
    PropositionSet,
    QueryAnalysis,
)


def _two_props() -> list[Proposition]:
    return [
        Proposition(
            quantifier="All",
            subject="M",
            predicate="P",
            source_evidence="a",
        ),
        Proposition(
            quantifier="All",
            subject="S",
            predicate="M",
            source_evidence="b",
        ),
    ]


def test_proposition_set_valid_indices():
    ps = PropositionSet(
        propositions=_two_props(),
        major_premise_index=0,
        minor_premise_index=1,
    )
    assert len(ps.propositions) == 2


def test_proposition_set_major_out_of_range():
    with pytest.raises(ValidationError):
        PropositionSet(
            propositions=_two_props(),
            major_premise_index=2,
            minor_premise_index=0,
        )


def test_proposition_set_minor_out_of_range():
    with pytest.raises(ValidationError):
        PropositionSet(
            propositions=_two_props(),
            major_premise_index=0,
            minor_premise_index=5,
        )


def test_proposition_set_same_index_rejected():
    with pytest.raises(ValidationError):
        PropositionSet(
            propositions=_two_props(),
            major_premise_index=0,
            minor_premise_index=0,
        )


def test_query_analysis_accepts_openai_style_json() -> None:
    payload = {
        "main_topic": "Socrates",
        "reasoning_type_required": ["Deductive", "Factual"],
        "key_terms": [
            {"term": "Socrates", "role": "Subject", "notes": "philosopher"},
            "plain",
        ],
        "expected_answer_format": {"type": "boolean", "examples": ["Yes"]},
    }
    qa = QueryAnalysis.model_validate(payload)
    assert "Deductive" in qa.reasoning_type
    assert "Socrates" in qa.key_terms[0]
    assert "plain" in qa.key_terms[1]
    assert "boolean" in qa.expected_answer_format


def test_evidence_collection_premises_and_numeric_confidence() -> None:
    ec = EvidenceCollection.model_validate(
        {
            "premises": [
                {
                    "fact": "All men are mortal.",
                    "relevance": 0.9,
                    "confidence": 0.95,
                }
            ],
            "query": "Is Socrates mortal?",
        }
    )
    assert ec.summary
    assert len(ec.evidence_items) == 1
    assert ec.evidence_items[0].relevance_score == 0.9
    assert ec.evidence_items[0].confidence == "high"


def test_evidence_item_statement_alias() -> None:
    ec = EvidenceCollection.model_validate(
        {
            "summary": "s",
            "evidence_items": [
                {
                    "statement": "All humans are mortal.",
                    "relevance": 0.9,
                    "confidence": 0.8,
                }
            ],
        }
    )
    assert ec.evidence_items[0].fact == "All humans are mortal."


def test_final_conclusion_alias_fields() -> None:
    fc = FinalConclusion.model_validate(
        {
            "final_conclusion": "Yes.",
            "logically_valid": True,
            "confidence": 0.9,
            "reasoning_summary": "Barbara.",
        }
    )
    assert fc.conclusion_text == "Yes."
    assert fc.is_valid is True


def test_final_conclusion_conclusion_key() -> None:
    fc = FinalConclusion.model_validate(
        {
            "conclusion": "Yes, Socrates is mortal.",
            "logically_valid": True,
            "confidence": 0.98,
            "reasoning_summary": "Syllogism.",
        }
    )
    assert fc.conclusion_text == "Yes, Socrates is mortal."
    assert fc.is_valid is True
