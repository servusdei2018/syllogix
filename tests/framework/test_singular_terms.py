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

from framework.engines.deductive import DeductiveEngine
from framework.models import Proposition, ReasoningStep, Syllogism
from framework.schemas import _parse_quantifier_form


def test_parse_quantifier_form_singulars():
    # Positive
    res1 = _parse_quantifier_form("Socrates is mortal")
    assert res1 == ("All", "Socrates", "mortal")

    # Negative
    res2 = _parse_quantifier_form("Socrates is not immortal")
    assert res2 == ("No", "Socrates", "immortal")


def test_proposition_string_representation(monkeypatch):
    # Mock nlp's is_singular_term to avoid depending on spacy model loaded in tests needlessly,
    # or just assume it works.
    monkeypatch.setattr("framework.nlp.is_singular_term", lambda x: True)

    p1 = Proposition("All", "Socrates", "mortal")
    assert str(p1) == "Socrates is mortal"

    p2 = Proposition("No", "Socrates", "immortal")
    assert str(p2) == "Socrates is not immortal"


def test_deductive_engine_applies_note_to_singular_term(monkeypatch):
    # Mock is_singular_subject so the engine observes we have a singular term
    monkeypatch.setattr(Proposition, "is_singular_subject", lambda self: True)

    # We must mock validate's nlp normalize wrapper if it uses it but we don't want spacy to load.
    # deductive.py calls self._terms_match which calls normalize_term.
    monkeypatch.setattr(
        "framework.engines.deductive.DeductiveEngine._terms_match",
        lambda self, t1, t2: t1.lower() == t2.lower(),
    )

    engine = DeductiveEngine()

    major = Proposition("All", "humans", "mortal")
    minor = Proposition("All", "Socrates", "humans")

    step = ReasoningStep(
        step_id=1,
        question="Is Socrates mortal?",
        reasoning_type="Deductive",
        syllogism=Syllogism(major_premise=major, minor_premise=minor),
    )

    validated_step = engine.validate(step)

    assert validated_step.is_valid is True
    assert validated_step.mood is not None
    assert (
        "Singular term treated as Universal class for logical validation"
        in validated_step.summary
    )
