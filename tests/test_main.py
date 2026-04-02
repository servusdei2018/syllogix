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

"""CLI smoke tests (no LLM calls)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import main as main_mod
from framework import FrameworkConfig

ROOT = Path(__file__).resolve().parents[1]


def test_public_framework_exports() -> None:
    import framework

    assert framework.__version__
    assert framework.LogicFramework
    assert framework.FrameworkConfig
    assert framework.load_framework_config_from_env


def test_main_py_help() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SYLLOGIX_PROVIDER" in r.stdout
    assert r.returncode == 0


def test_build_parser_query_optional() -> None:
    p = main_mod.build_parser()
    args = p.parse_args(["hello world"])
    assert args.query == "hello world"
    assert args.file is None


def test_main_invokes_reason_sync(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = MagicMock()
    chain.__str__ = MagicMock(return_value="--- report ---")
    mock_fw = MagicMock()
    mock_fw.reason_sync = MagicMock(return_value=chain)

    def stub_logic_framework(_cfg: FrameworkConfig | None) -> MagicMock:
        return mock_fw

    def stub_load_config() -> FrameworkConfig:
        return FrameworkConfig()

    monkeypatch.setattr(main_mod, "LogicFramework", stub_logic_framework)
    monkeypatch.setattr(main_mod, "load_framework_config_from_env", stub_load_config)
    monkeypatch.setattr(sys, "argv", ["syllogix", "test query"])
    main_mod.main()
    mock_fw.reason_sync.assert_called_once_with("test query")
    out = capsys.readouterr().out
    assert "--- report ---" in out
