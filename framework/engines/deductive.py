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

from typing import Callable, Dict

from ..models import Proposition, ReasoningStep


class DeductiveEngine:
    """
    Validates a ReasoningStep against the classical deductive moods (syllogisms).

    This engine performs purely symbolic, deductive validation of syllogistic
    premises. It intentionally does not perform or label inductive reasoning:
    statistical or hypothesis-driven evidence should be surfaced to this
    engine as Observations (handled upstream by the formalizer / RAG layer).

    Legend:
    S = Subject of the Conclusion
    P = Predicate of the Conclusion
    M = Middle Term (in both premises, not in conclusion)

    Quantifiers:
    A = All (Universal Affirmative)
    E = No (Universal Negative)
    I = Some (Particular Affirmative)
    O = Some...not (Particular Negative)
    """

    def __init__(self):
        # Map of classical deductive moods to their validator callables.
        # Each validator accepts (major: Proposition, minor: Proposition)
        # and returns a `Proposition | None`. Only deductive moods are
        # registered here (no inductive/statistical moods).
        self.mood_validators: Dict[
            str, Callable[[Proposition, Proposition], Proposition | None]
        ] = {
            # Figure 1
            "Barbara": self._validate_barbara,  # AAA-1
            "Celarent": self._validate_celarent,  # EAE-1
            "Darii": self._validate_darii,  # AII-1
            "Ferio": self._validate_ferio,  # EIO-1
            # Figure 2
            "Cesare": self._validate_cesare,  # EAE-2
            "Camestres": self._validate_camestres,  # AEE-2
            "Festino": self._validate_festino,  # EIO-2
            "Baroco": self._validate_baroco,  # AOO-2
            # Figure 3
            "Darapti": self._validate_darapti,  # AAI-3
            "Felapton": self._validate_felapton,  # EAO-3
            "Disamis": self._validate_disamis,  # IAI-3
            "Datisi": self._validate_datisi,  # AII-3
            "Bocardo": self._validate_bocardo,  # OAO-3
            "Ferison": self._validate_ferison,  # EIO-3
            # Figure 4
            "Baralipton": self._validate_baralipton,  # AAI-4
            "Celantes": self._validate_celantes,  # EAE-4
            "Dabitis": self._validate_dabitis,  # IAI-4
            "Fapesmo": self._validate_fapesmo,  # EAO-4
        }

    def validate(self, step: ReasoningStep) -> ReasoningStep:
        """
        Tries to find a valid deductive mood that matches
        the premises provided by the LLM.
        """
        major = step.syllogism.major_premise if step.syllogism else None
        minor = step.syllogism.minor_premise if step.syllogism else None

        if not major or not minor:
            step.is_valid = False
            step.summary += " [EngineError: Missing one or more premises]"
            return step

        for mood, validator_func in self.mood_validators.items():
            conclusion = validator_func(major, minor)
            if conclusion:
                step.is_valid = True
                if step.syllogism is not None:
                    step.syllogism.conclusion = conclusion
                step.mood = mood
                step.confidence = 1.0  # Deductive logic is 100% confident
                return step

        step.is_valid = False
        step.summary += " [EngineError: Premises do not form a valid known syllogism]"
        return step

    # --- FIGURE 1: M-P, S-M ---

    def _validate_barbara(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """AAA-1: All M are P, All S are M -> All S are P"""
        if (
            major.quantifier == "All"
            and minor.quantifier == "All"
            and major.subject == minor.predicate
        ):  # M-P, S-M
            S = minor.subject
            P = major.predicate
            return Proposition(quantifier="All", subject=S, predicate=P)
        return None

    def _validate_celarent(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """EAE-1: No M are P, All S are M -> No S are P"""
        if (
            major.quantifier == "No"
            and minor.quantifier == "All"
            and major.subject == minor.predicate
        ):  # M-P, S-M
            S = minor.subject
            P = major.predicate
            return Proposition(quantifier="No", subject=S, predicate=P)
        return None

    # --- FIGURE 2: P-M, S-M ---

    def _validate_cesare(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """EAE-2: No P are M, All S are M -> No S are P"""
        if (
            major.quantifier == "No"
            and minor.quantifier == "All"
            and major.predicate == minor.predicate
        ):  # P-M, S-M
            S = minor.subject
            P = major.subject
            return Proposition(quantifier="No", subject=S, predicate=P)
        return None

    def _validate_darii(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """AII-1: All M are P, Some S are M -> Some S are P"""
        if (
            major.quantifier == "All"
            and minor.quantifier == "Some"
            and major.subject == minor.predicate
        ):  # M-P, S-M (minor particular)
            S = minor.subject
            P = major.predicate
            return Proposition(quantifier="Some", subject=S, predicate=P)
        return None

    def _validate_ferio(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """EIO-1: No M are P, Some S are M -> Some S are not P"""
        if (
            major.quantifier == "No"
            and minor.quantifier == "Some"
            and major.subject == minor.predicate
        ):  # M-P, S-M (minor particular)
            S = minor.subject
            P = major.predicate
            return Proposition(quantifier="Some...not", subject=S, predicate=P)
        return None

    def _validate_camestres(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """AEE-2: All P are M, No S are M -> No S are P"""
        if (
            major.quantifier == "All"
            and minor.quantifier == "No"
            and major.predicate == minor.predicate
        ):  # P-M, S-M (shared M)
            S = minor.subject
            P = major.subject
            return Proposition(quantifier="No", subject=S, predicate=P)
        return None

    def _validate_festino(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """EIO-2: No P are M, Some S are M -> Some S are not P"""
        if (
            major.quantifier == "No"
            and minor.quantifier == "Some"
            and major.predicate == minor.predicate
        ):  # P-M, S-M
            S = minor.subject
            P = major.subject
            return Proposition(quantifier="Some...not", subject=S, predicate=P)
        return None

    def _validate_baroco(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """AOO-2: All P are M, Some S are not M -> Some S are not P"""
        if (
            major.quantifier == "All"
            and minor.quantifier == "Some...not"
            and major.predicate == minor.predicate
        ):  # P-M, S-M (minor particular negative)
            S = minor.subject
            P = major.subject
            return Proposition(quantifier="Some...not", subject=S, predicate=P)
        return None

    # --- FIGURE 3: M-P, M-S ---

    def _validate_darapti(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """AAI-3: All M are P, All M are S -> Some S are P"""
        if (
            major.quantifier == "All"
            and minor.quantifier == "All"
            and major.subject == minor.subject
        ):  # M-P, M-S
            S = minor.predicate  # M are S -> S is minor.predicate
            P = major.predicate
            return Proposition(quantifier="Some", subject=S, predicate=P)
        return None

    def _validate_felapton(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """EAO-3: No M are P, All M are S -> Some S are not P"""
        if (
            major.quantifier == "No"
            and minor.quantifier == "All"
            and major.subject == minor.subject
        ):  # M-P, M-S
            S = minor.predicate
            P = major.predicate
            return Proposition(quantifier="Some...not", subject=S, predicate=P)
        return None

    def _validate_disamis(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """IAI-3: Some M are P, All M are S -> Some S are P"""
        if (
            major.quantifier == "Some"
            and minor.quantifier == "All"
            and major.subject == minor.subject
        ):  # Some M are P (particular), All M are S
            S = minor.predicate
            P = major.predicate
            return Proposition(quantifier="Some", subject=S, predicate=P)
        return None

    def _validate_datisi(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """AII-3: All M are P, Some M are S -> Some S are P"""
        if (
            major.quantifier == "All"
            and minor.quantifier == "Some"
            and major.subject == minor.subject
        ):  # All M are P, Some M are S (particular)
            S = minor.predicate
            P = major.predicate
            return Proposition(quantifier="Some", subject=S, predicate=P)
        return None

    def _validate_bocardo(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """OAO-3: Some M are not P, All M are S -> Some S are not P"""
        if (
            major.quantifier == "Some...not"
            and minor.quantifier == "All"
            and major.subject == minor.subject
        ):  # Some M are not P, All M are S
            S = minor.predicate
            P = major.predicate
            return Proposition(quantifier="Some...not", subject=S, predicate=P)
        return None

    def _validate_ferison(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """EIO-3: No M are P, Some M are S -> Some S are not P"""
        if (
            major.quantifier == "No"
            and minor.quantifier == "Some"
            and major.subject == minor.subject
        ):  # No M are P, Some M are S
            S = minor.predicate
            P = major.predicate
            return Proposition(quantifier="Some...not", subject=S, predicate=P)
        return None

    # --- FIGURE 4: P-M, M-S ---

    def _validate_baralipton(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """AAI-4: All P are M, All M are S -> Some S are P"""
        if (
            major.quantifier == "All"
            and minor.quantifier == "All"
            and major.predicate == minor.subject
        ):  # P-M, M-S
            S = minor.predicate
            P = major.subject
            return Proposition(quantifier="Some", subject=S, predicate=P)
        return None

    def _validate_celantes(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """EAE-4: No P are M, All M are S -> No S are P"""
        if (
            major.quantifier == "No"
            and minor.quantifier == "All"
            and major.predicate == minor.subject
        ):  # No P are M, All M are S
            S = minor.predicate
            P = major.subject
            return Proposition(quantifier="No", subject=S, predicate=P)
        return None

    def _validate_dabitis(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """IAI-4: Some P are M, All M are S -> Some S are P"""
        if (
            major.quantifier == "Some"
            and minor.quantifier == "All"
            and major.predicate == minor.subject
        ):  # Some P are M, All M are S
            S = minor.predicate
            P = major.subject
            return Proposition(quantifier="Some", subject=S, predicate=P)
        return None

    def _validate_fapesmo(
        self, major: Proposition, minor: Proposition
    ) -> Proposition | None:
        """EAO-4: No P are M, Some M are S -> Some S are not P"""
        if (
            major.quantifier == "No"
            and minor.quantifier == "Some"
            and major.predicate == minor.subject
        ):  # No P are M, Some M are S
            S = minor.predicate
            P = major.subject
            return Proposition(quantifier="Some...not", subject=S, predicate=P)
        return None
