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


"""
Pytest configuration.
"""

import os
import sys
from pathlib import Path

import pytest

# pytest-asyncio configuration
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the session."""
    import asyncio

    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


# Determine the project root as the parent directory of the `tests` directory.
# Path(__file__).resolve() -> .../syllogix/tests/conftest.py
# .parents[0] -> .../syllogix/tests
# .parents[1] -> .../syllogix  (project root)
_project_root = Path(__file__).resolve().parents[1]
_project_root_str = str(_project_root)

# Insert project root at the front of sys.path so it takes precedence over
# installed packages or other paths. This mirrors what setting PYTHONPATH
# to the project root would achieve when running tests locally.
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Also ensure PYTHONPATH environment contains the project root for any
# subprocesses that may be spawned during tests.
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _project_root_str not in _existing_pythonpath.split(os.pathsep):
    if _existing_pythonpath:
        os.environ["PYTHONPATH"] = _project_root_str + os.pathsep + _existing_pythonpath
    else:
        os.environ["PYTHONPATH"] = _project_root_str
