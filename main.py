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

"""CLI: run syllogistic reasoning. Full env documentation: README.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from framework import LogicFramework, load_framework_config_from_env

_EPILOG = """Environment (optional unless you call the API; see README.md):
  SYLLOGIX_PROVIDER, SYLLOGIX_MODEL, SYLLOGIX_API_KEY
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY
  SYLLOGIX_BASE_URL, SYLLOGIX_TIMEOUT, SYLLOGIX_MAX_RETRIES
  SYLLOGIX_ENABLE_CACHING, SYLLOGIX_CACHE_TTL, SYLLOGIX_LOG_LEVEL
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="syllogix",
        description="Run agentic syllogistic reasoning for a query (LLM API required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question to reason about (optional if using --file or stdin)",
    )
    p.add_argument(
        "-f",
        "--file",
        metavar="PATH",
        default=None,
        help="Read query text from this file instead of arguments or stdin",
    )
    return p


def _resolve_query(args: argparse.Namespace) -> str:
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8").strip()
    if args.query:
        return args.query.strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    text = _resolve_query(args)
    if not text:
        parser.error("provide a query argument, --file PATH, or pipe text on stdin")

    try:
        fw = LogicFramework(load_framework_config_from_env())
        chain = fw.reason_sync(text)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e

    print(chain)


if __name__ == "__main__":
    main()
