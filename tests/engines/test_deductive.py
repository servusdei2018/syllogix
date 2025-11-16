# Syllogix
# Copyright (C) 2025  Nathanael Bracy
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

from framework.engines.deductive import DeductiveEngine
from framework.models import Proposition, ReasoningStep, Syllogism


@pytest.mark.parametrize(
    "mood, major, minor, expected_quant, expected_subject, expected_predicate",
    [
        # Figure 1 (M-P, S-M)
        (
            "Barbara",
            Proposition(quantifier="All", subject="M", predicate="P"),
            Proposition(quantifier="All", subject="S", predicate="M"),
            "All",
            "S",
            "P",
        ),
        (
            "Celarent",
            Proposition(quantifier="No", subject="M", predicate="P"),
            Proposition(quantifier="All", subject="S", predicate="M"),
            "No",
            "S",
            "P",
        ),
        (
            "Darii",
            Proposition(quantifier="All", subject="M", predicate="P"),
            Proposition(quantifier="Some", subject="S", predicate="M"),
            "Some",
            "S",
            "P",
        ),
        (
            "Ferio",
            Proposition(quantifier="No", subject="M", predicate="P"),
            Proposition(quantifier="Some", subject="S", predicate="M"),
            "Some...not",
            "S",
            "P",
        ),
        # Figure 2 (P-M, S-M)
        (
            "Cesare",
            Proposition(quantifier="No", subject="P", predicate="M"),
            Proposition(quantifier="All", subject="S", predicate="M"),
            "No",
            "S",
            "P",
        ),
        (
            "Camestres",
            Proposition(quantifier="All", subject="P", predicate="M"),
            Proposition(quantifier="No", subject="S", predicate="M"),
            "No",
            "S",
            "P",
        ),
        (
            "Festino",
            Proposition(quantifier="No", subject="P", predicate="M"),
            Proposition(quantifier="Some", subject="S", predicate="M"),
            "Some...not",
            "S",
            "P",
        ),
        (
            "Baroco",
            Proposition(quantifier="All", subject="P", predicate="M"),
            Proposition(quantifier="Some...not", subject="S", predicate="M"),
            "Some...not",
            "S",
            "P",
        ),
        # Figure 3 (M-P, M-S)
        (
            "Darapti",
            Proposition(quantifier="All", subject="M", predicate="P"),
            Proposition(quantifier="All", subject="M", predicate="S"),
            "Some",
            "S",
            "P",
        ),
        (
            "Felapton",
            Proposition(quantifier="No", subject="M", predicate="P"),
            Proposition(quantifier="All", subject="M", predicate="S"),
            "Some...not",
            "S",
            "P",
        ),
        (
            "Disamis",
            Proposition(quantifier="Some", subject="M", predicate="P"),
            Proposition(quantifier="All", subject="M", predicate="S"),
            "Some",
            "S",
            "P",
        ),
        (
            "Datisi",
            Proposition(quantifier="All", subject="M", predicate="P"),
            Proposition(quantifier="Some", subject="M", predicate="S"),
            "Some",
            "S",
            "P",
        ),
        (
            "Bocardo",
            Proposition(quantifier="Some...not", subject="M", predicate="P"),
            Proposition(quantifier="All", subject="M", predicate="S"),
            "Some...not",
            "S",
            "P",
        ),
        (
            "Ferison",
            Proposition(quantifier="No", subject="M", predicate="P"),
            Proposition(quantifier="Some", subject="M", predicate="S"),
            "Some...not",
            "S",
            "P",
        ),
        # Figure 4 (P-M, M-S)
        (
            "Baralipton",
            Proposition(quantifier="All", subject="P", predicate="M"),
            Proposition(quantifier="All", subject="M", predicate="S"),
            "Some",
            "S",
            "P",
        ),
        (
            "Celantes",
            Proposition(quantifier="No", subject="P", predicate="M"),
            Proposition(quantifier="All", subject="M", predicate="S"),
            "No",
            "S",
            "P",
        ),
        (
            "Dabitis",
            Proposition(quantifier="Some", subject="P", predicate="M"),
            Proposition(quantifier="All", subject="M", predicate="S"),
            "Some",
            "S",
            "P",
        ),
        (
            "Fapesmo",
            Proposition(quantifier="No", subject="P", predicate="M"),
            Proposition(quantifier="Some", subject="M", predicate="S"),
            "Some...not",
            "S",
            "P",
        ),
    ],
)
def test_deductive_engine_recognizes_moods(
    mood, major, minor, expected_quant, expected_subject, expected_predicate
):
    """
    For each mood, construct a syllogism that matches the mood's structural pattern and verify:
      - The engine marks the step valid
      - The detected mood matches the expected one
      - The produced conclusion has the expected quantifier/subject/predicate
    """
    engine = DeductiveEngine()
    syll = Syllogism(major_premise=major, minor_premise=minor, conclusion=None)
    step = ReasoningStep(step_id=1, question="Test", reasoning_type="Deductive")
    step.syllogism = syll

    validated = engine.validate(step)

    assert validated.is_valid is True, f"Engine failed to validate mood {mood}"
    assert validated.mood == mood, (
        f"Expected mood {mood}, got {validated.mood} for major={major} minor={minor}"
    )
    assert validated.syllogism is not None
    assert validated.syllogism.conclusion is not None, (
        "Conclusion should be set for valid syllogism"
    )

    concl = validated.syllogism.conclusion
    assert concl.quantifier == expected_quant
    assert concl.subject == expected_subject
    assert concl.predicate == expected_predicate


def test_deductive_engine_rejects_invalid_premises():
    """
    If premises do not form any supported syllogism, engine should
    mark the step invalid and not fabricate a conclusion.
    """
    engine = DeductiveEngine()

    # Premises that don't share a middle term or don't match any figure/mood
    major = Proposition(quantifier="All", subject="X", predicate="Y")
    minor = Proposition(quantifier="All", subject="A", predicate="B")
    syll = Syllogism(major_premise=major, minor_premise=minor, conclusion=None)
    step = ReasoningStep(step_id=2, question="Invalid", reasoning_type="Deductive")
    step.syllogism = syll

    validated = engine.validate(step)

    assert validated.is_valid is False
    assert validated.syllogism is not None
    assert validated.syllogism.conclusion is None
    assert (
        "Premises do not form a valid known syllogism" in validated.summary
        or "EngineError" in validated.summary
    )


def test_deductive_engine_handles_missing_premises():
    """
    If the step lacks one or both premises, engine should mark it invalid and
    include an engine error marker in the summary.
    """
    engine = DeductiveEngine()

    # Completely missing syllogism
    step_none = ReasoningStep(step_id=3, question="Missing", reasoning_type="Deductive")
    step_none.syllogism = None

    validated_none = engine.validate(step_none)
    assert validated_none.is_valid is False
    assert "[EngineError: Missing" in validated_none.summary

    # Syllogism present but missing major/minor (malformed)
    partial_syll = Syllogism(major_premise=None, minor_premise=None, conclusion=None)
    step_partial = ReasoningStep(
        step_id=4, question="Partial", reasoning_type="Deductive"
    )
    step_partial.syllogism = partial_syll

    validated_partial = engine.validate(step_partial)
    assert validated_partial.is_valid is False
    assert "[EngineError: Missing" in validated_partial.summary
