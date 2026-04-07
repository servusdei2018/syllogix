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
uv run main.py "Is Socrates mortal?"

# Output:

 ============================================================
   REASONING CHAIN: 'Is Socrates mortal?'
 ============================================================

  Step 1: Query Analysis
  ------------------------------------------------------------
  Type      : Observation
  Status    : ✓ VALID  (confidence: 90.0%)
  Summary   : Topic: Socrates' mortality (a property attribution question about an individual). Type: Deductive reasoning (syllogistic/class-inclusion inference) if premises are provided or assumed, e.g., 'All humans are mortal' and 'Socrates is human'.; Factual/encyclopedic reasoning if treated as a historical-biographical question about the real person Socrates.

  Step 2: Evidence Retrieval
  ------------------------------------------------------------
  Type      : Observation
  Status    : ✓ VALID  (confidence: 80.0%)
  Summary   : Socrates' mortality (a property attribution question about an individual)
  Evidence  : 6 item(s)
    [ev_0] Socrates was a human being (an Athenian man and philosopher who lived in the 5th century BCE).
    [ev_1] All human beings are mortal (humans eventually die).
    [ev_2] Ancient sources report that Socrates died in 399 BCE after being sentenced to death in Athens and drinking hemlock.
    [ev_3] Plato's dialogues (e.g., the 'Phaedo') depict Socrates' execution and death by hemlock as the conclusion of his trial an
    [ev_4] Xenophon's writings (e.g., 'Apology of Socrates') also describe Socrates being condemned and put to death in Athens.
    [ev_5] Socrates is treated in historical scholarship as a historical person rather than a mythical immortal figure, and no cred

  Step 3: Proposition Formation
  ------------------------------------------------------------
  Type      : Observation
  Status    : ✓ VALID  (confidence: 70.0%)
  Summary   : Formed 4 propositions
  Major     : All human beings are mortal
  Minor     : Some Socrates are human being

  Step 4: Deductive Reasoning
  ------------------------------------------------------------
  Type      : Deductive
  Status    : ✗ INVALID  (confidence: 0.0%)
  Summary   : [EngineError: Premises do not form a valid known syllogism]

Invalid as stated because the minor premise is malformed: 'Some Socrates are human being' treats 'Socrates' as a class with multiple members (an I-proposition), but in the traditional syllogistic form 'Socrates' is a singular term. Without a coherent minor premise in standard categorical form, no valid syllogism/mood can be identified. If corrected to 'All Socrates are human beings' (or 'Socrates is a human being'), then with the major premise it yields a valid Barbara syllogism with conclusion 'All Socrates are mortal' (or 'Socrates is mortal').
  Major     : All human beings are mortal
  Minor     : Some Socrates are human being

  Step 5: Final Conclusion
  ------------------------------------------------------------
  Type      : Observation
  Status    : ✓ VALID  (confidence: 90.0%)
  Summary   : Socrates is mortal.

============================================================
  FINAL CONCLUSION
============================================================
  Socrates is mortal.
============================================================
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
