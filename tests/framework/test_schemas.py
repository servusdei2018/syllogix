# Syllogix
# Copyright (C) 2026  Nathanael Bracy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest
from pydantic import ValidationError

from framework.schemas import Proposition, PropositionSet


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
