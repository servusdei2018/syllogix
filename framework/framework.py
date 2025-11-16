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

from .engines.deductive import DeductiveEngine
from .models import ReasoningChain


class LogicFramework:
    """Orchestrates the entire reasoning process, from initial query to a final, auditable ReasoningChain."""

    def __init__(self):
        self.deductive_engine = DeductiveEngine()

    def reason(self, query: str) -> ReasoningChain:
        """
        Main entry point. Executes the full chain of logic.
        """

        return ReasoningChain(main_query=query)
