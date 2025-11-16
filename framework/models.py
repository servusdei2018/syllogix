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


from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RAGSource:
    """A single piece of retrieved evidence."""

    source_id: str
    text: str
    url: str | None = None


Quantifier = Literal[
    "All",  # Universal Affirmative
    "No",  # Universal Negative
    "Some",  # Particular Affirmative
    "Some...not",  # Particular Negative
    "Statistical",  # e.g., "80% of"
]


@dataclass
class Proposition:
    """A single, structured logical proposition."""

    quantifier: Quantifier
    subject: str
    predicate: str
    is_negated: bool = False  # For complex forms

    # For "Statistical" quantifier
    stat_value: dict[str, Any] | None = (
        None  # e.g., {"type": "percentage", "value": 80}
    )

    def __str__(self):
        if self.quantifier == "Statistical":
            if not self.stat_value:
                raise ValueError("Statistical propositions must have a stat_value.")
            val = self.stat_value.get("value", "Most")
            return f"{val}% of {self.subject} are {self.predicate}"
        if self.quantifier == "Some...not":
            return f"Some {self.subject} are not {self.predicate}"
        return f"{self.quantifier} {self.subject} are {self.predicate}"


ReasoningType = Literal[
    "Deductive",  # Standard Deduction
    "Observation",  # A starting fact
]


@dataclass
class Syllogism:
    """A structured syllogism with major and minor premises."""

    major_premise: Proposition | None = None
    minor_premise: Proposition | None = None
    conclusion: Proposition | None = None


@dataclass
class ReasoningStep:
    """One complete, validated link in the reasoning chain."""

    step_id: int
    question: str
    reasoning_type: ReasoningType

    # Evidence
    rag_sources: list[RAGSource] = field(default_factory=list)

    # Formalized Logic
    syllogism: Syllogism | None = None

    # Verdict
    mood: str | None = None  # e.g., "Barbara", "StatisticalSyllogism"
    is_valid: bool = False
    confidence: float = 1.0  # 1.0 for valid deduction, < 1.0 for induction

    # Natural language summary of the step
    summary: str = ""


@dataclass
class ReasoningChain:
    """The complete, final report of the framework's thought process."""

    main_query: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_conclusion_summary: str | None = None

    def add_step(self, step: ReasoningStep):
        """Adds a new step to the chain."""
        self.steps.append(step)

    def get_proven_premise(self, subject: str) -> Proposition | None:
        """
        Finds a valid conclusion from earlier in the chain to
        use as a new, trusted premise.

        Behavior:
          - Searches previous steps in reverse (most recent first) for a valid
            conclusion.
          - If a conclusion's subject matches the requested subject, returns
            that conclusion directly.
          - If a conclusion's predicate matches the requested subject, attempts
            a safe conversion when logically valid:
              * "No P are Q"  <-> "No Q are P"  (E: symmetric)  -> safe to invert
              * "Some P are Q" <-> "Some Q are P" (I: symmetric) -> safe to invert
            Other conversions (e.g., turning "All" into "All" by swapping) are
            NOT performed because they are not logically valid without
            additional existential assumptions.
        """
        if subject is None:
            return None

        target = subject.strip().lower()

        for step in reversed(self.steps):
            if not (step.is_valid and step.syllogism and step.syllogism.conclusion):
                continue

            concl = step.syllogism.conclusion

            if concl.subject and concl.subject.strip().lower() == target:
                return concl

            if concl.predicate and concl.predicate.strip().lower() == target:
                quant = concl.quantifier

                # E (No ...) is symmetric: "No P are Q" -> "No Q are P"
                if quant == "No":
                    return Proposition(
                        quantifier="No", subject=subject, predicate=concl.subject
                    )

                # I (Some ...) is symmetric: "Some P are Q" -> "Some Q are P"
                if quant == "Some":
                    return Proposition(
                        quantifier="Some", subject=subject, predicate=concl.subject
                    )

                # For "Some...not" (O), "All" (A), "Statistical" etc. we avoid converting
                # because those inversions are not guaranteed valid without extra
                # assumptions (existential import, statistical semantics, etc.).
                # Continue searching older steps if available.
                continue

        return None

    def __str__(self):
        """Pretty-prints the entire reasoning chain."""
        report = f"--- Reasoning Chain for: '{self.main_query}' ---\n"
        for step in self.steps:
            report += f"\nStep {step.step_id}: {step.question}\n"
            report += f"  Type: {step.reasoning_type} (Mood: {step.mood})\n"
            report += f"  Status: {'VALID' if step.is_valid else 'INVALID'} (Conf: {step.confidence * 100}%)\n"
            if step.syllogism:
                if step.syllogism.major_premise:
                    report += f"  Major: {step.syllogism.major_premise}\n"
                if step.syllogism.minor_premise:
                    report += f"  Minor: {step.syllogism.minor_premise}\n"
                if step.syllogism.conclusion:
                    report += f"  Conclusion: {step.syllogism.conclusion}\n"
            if step.rag_sources:
                report += f"  Evidence: {', '.join([s.source_id for s in step.rag_sources])}\n"
        report += f"\n--- Final Conclusion ---\n{self.final_conclusion_summary}\n"
        return report
