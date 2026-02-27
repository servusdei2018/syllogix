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

from framework.models import (
    Proposition,
    Quantifier,
    ReasoningChain,
    ReasoningStep,
    Syllogism,
)


def _make_step_with_conclusion(
    step_id: int, conclusion: Proposition, valid: bool = True
):
    step = ReasoningStep(
        step_id=step_id, question=f"Q{step_id}", reasoning_type="Deductive"
    )
    step.syllogism = Syllogism(
        major_premise=None, minor_premise=None, conclusion=conclusion
    )
    step.is_valid = valid
    return step


def test_get_proven_premise_direct_subject_match_returns_conclusion():
    """
    If a previous valid step concluded with a proposition whose subject matches
    the requested subject, that exact proposition should be returned.
    """
    chain = ReasoningChain(main_query="Does this work?")

    concl = Proposition(
        quantifier="Some",
        subject="Cats",
        predicate="Mammals",
    )
    step = _make_step_with_conclusion(1, concl, valid=True)
    chain.add_step(step)

    found = chain.get_proven_premise("Cats")

    # Should return the exact conclusion object placed into the step
    assert found is not None
    assert step.syllogism is not None
    assert found is step.syllogism.conclusion
    assert found.subject == "Cats"
    assert found.predicate == "Mammals"
    assert found.quantifier == "Some"


@pytest.mark.parametrize(
    "quant, orig_subject, orig_predicate, query_subject, exp_quant, exp_subject, exp_predicate",
    [
        # E (No): conversion allowed -> swap subject/predicate
        ("No", "Dogs", "Animals", "Animals", "No", "Animals", "Dogs"),
        # I (Some): conversion allowed -> swap subject/predicate
        ("Some", "birds", "creatures", "Creatures", "Some", "Creatures", "birds"),
    ],
)
def test_get_proven_premise_predicate_based_conversion_allowed(
    quant: Quantifier,
    orig_subject: str,
    orig_predicate: str,
    query_subject: str,
    exp_quant: Quantifier,
    exp_subject: str,
    exp_predicate: str,
) -> None:
    """
    If a conclusion's predicate matches the requested subject and the quantifier
    permits simple conversion (E or I), get_proven_premise should return a new
    Proposition with subject==query_subject and predicate==original subject.
    """
    chain = ReasoningChain(main_query="Conversion test")

    conclusion = Proposition(
        quantifier=quant,
        subject=orig_subject,
        predicate=orig_predicate,
    )
    step = _make_step_with_conclusion(1, conclusion, valid=True)
    chain.add_step(step)

    found = chain.get_proven_premise(query_subject)
    assert found is not None
    assert found.quantifier == exp_quant
    assert found.subject == exp_subject
    assert found.predicate == exp_predicate


def test_get_proven_premise_skips_invalid_conversions_and_falls_back_or_none():
    """
    If a conclusion's predicate matches the requested subject but the quantifier
    is not convertible (e.g., 'All', 'Some...not', 'Statistical'), it should be
    skipped and the search should continue. If an older valid conclusion exists
    with a matching subject, it should be returned. If none exists, return None.
    """
    chain = ReasoningChain(main_query="Skip conversion test")

    # Older valid direct match (should be returned if later non-convertible step is skipped)
    direct_concl = Proposition(
        quantifier="Some", subject="Plants", predicate="Organisms"
    )
    older_step = _make_step_with_conclusion(1, direct_concl, valid=True)
    chain.add_step(older_step)

    # Newer conclusion that would match by predicate but has a non-convertible quantifier
    non_convertible = Proposition(quantifier="All", subject="X", predicate="Plants")
    newer_step = _make_step_with_conclusion(2, non_convertible, valid=True)
    chain.add_step(newer_step)

    # Because the newer step cannot be converted, the function should continue and
    # return the older direct-match conclusion.
    found = chain.get_proven_premise("Plants")
    assert found is not None
    assert found.subject == "Plants"
    assert found.predicate == "Organisms"

    # When only a non-convertible matching-predicate exists -> should return None
    chain2 = ReasoningChain(main_query="Only non-convertible")
    only_step = _make_step_with_conclusion(
        1, Proposition(quantifier="Some...not", subject="A", predicate="B"), valid=True
    )
    chain2.add_step(only_step)

    assert chain2.get_proven_premise("B") is None

    chain3 = ReasoningChain(main_query="Statistical non-convertible")
    stat_step = _make_step_with_conclusion(
        1,
        Proposition(
            quantifier="Statistical",
            subject="Group1",
            predicate="Group2",
        ),
        valid=True,
    )
    chain3.add_step(stat_step)
    assert chain3.get_proven_premise("Group2") is None
