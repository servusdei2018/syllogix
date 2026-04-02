# syllogix - agentic syllogistic reasoning

This framework facilitates Large Language Models (LLMs) ability to "think" using a verifiable, step-by-step logical process. Instead of producing a single, black-box answer, it generates an auditable train of thought that validates each conclusion against demonstrably sound deductive and inductive logic. The final output is not just an answer, but a "proof" of how that answer was reached.

## Install and run

From the repository root, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run a query (requires a configured API key for your chosen provider):

```bash
uv run syllogix "Is Socrates mortal?"
# or
uv run python main.py "Is Socrates mortal?"
```

Read the query from a file:

```bash
uv run syllogix --file question.txt
```

Pipe stdin when you do not pass a positional query:

```bash
echo "Your question here" | uv run syllogix
```

For an editable install with another tool: `pip install -e .` (same CLI entry `syllogix`).

## Configuration (environment variables)

All variables are optional unless you call an LLM API; you must supply an API key for real runs.

| Variable | Role |
|----------|------|
| `SYLLOGIX_PROVIDER` | LLM provider id (default `openai`). |
| `SYLLOGIX_MODEL` | Model name (default `gpt-5.2`). |
| `SYLLOGIX_API_KEY` | API key for any provider (preferred generic). |
| `OPENAI_API_KEY` | Used when `SYLLOGIX_API_KEY` is unset and provider is OpenAI. |
| `ANTHROPIC_API_KEY` | Used when provider is Anthropic. |
| `GOOGLE_API_KEY` | Used when provider is Google. |
| `OPENROUTER_API_KEY` | Used when provider is OpenRouter. |
| `SYLLOGIX_BASE_URL` | Optional provider base URL. |
| `SYLLOGIX_TIMEOUT` | Request timeout in seconds (default `30`). |
| `SYLLOGIX_MAX_RETRIES` | Max retries (default `3`). |
| `SYLLOGIX_ENABLE_CACHING` | `true` / `false` / `1` / `0` (default true). |
| `SYLLOGIX_CACHE_TTL` | Cache TTL in seconds (default `3600`). |
| `SYLLOGIX_LOG_LEVEL` | Logging level name (default `INFO`). |

`FrameworkConfig` also defines `max_reasoning_steps` and `min_confidence_threshold`; they are not yet read from the environment (see project status notes).

## Library usage

```python
from framework import LogicFramework, load_framework_config_from_env

fw = LogicFramework(load_framework_config_from_env())
chain = fw.reason_sync("Your question")
print(chain)
```

## License

```
Syllogix
Copyright (C) 2025  Nathanael Bracy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
