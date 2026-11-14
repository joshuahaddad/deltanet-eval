# Build: filing-recall, a small long-context extraction eval

## Context

This repo goes with a blog post on linear attention (DeltaNet, Gated DeltaNet, Kimi Delta Attention). The post's claim for engineers: models with a fixed-size recurrent state, such as Kimi Linear (a hybrid of 3 KDA layers to 1 full-attention layer), are cheap at long context and good at *tolerant* recall (gist, summaries), while *exact* recall (IDs, figures, dates) is the weak spot. This app tests that claim on real documents, and it's also a usable extraction tool.

## Where it goes

This repo also holds the blog post's material: `README.md`, `reference.py`, `animations/`, `figures/`, `interactive/`, `qa/` and the root `pyproject.toml`. **Build everything inside `eval/` and don't change anything outside it**, except for adding app-specific lines to the root `.gitignore`. `eval/` gets its own `pyproject.toml`, `uv.lock`, README and tests, and nothing in it may import from the repo root. That way it can later be split into its own repo with `git subtree split --prefix eval`. Commands below run from `eval/`.

## What to build

A local webapp with a Python backend (FastAPI, managed with uv) and a single HTML page. Keep it to as few files as possible.

**Documents:** public SEC 10-K filings, fetched from EDGAR as HTML by ticker and fiscal year. Convert them to clean text and keep tables readable. Don't convert from PDF. Send EDGAR the User-Agent it requires, using a generic app name and a contact read from an env var.

**Ground truth:**
- *Exact fields* come from the filing's XBRL facts for one stated fiscal year: revenue, net income, total assets, shares outstanding, fiscal year end date, and a few more. Each question names the fiscal year.
- *Tolerant fields* are short questions about the narrative sections, such as "What are the two main risk factors?" or "What does the company say drives revenue growth?". An LLM judge scores them against the relevant passage.
- *Planted facts* give control over depth. Insert synthetic facts at 10/30/50/70/90% of the way through the document. Use made-up exact values (for example "Internal reference code: QX-48213-7") and short tolerant statements.

**Eval mode:** truncate each filing to 8k, 32k and 128k tokens (and the full length if it fits the model). Ask the model to extract every field as JSON. Score:
- exact fields by match after normalising numbers (units, scale such as "in millions", commas, currency signs) and dates
- tolerant fields with the judge

Show heatmaps of accuracy by depth and length, separately for exact and tolerant fields and for real and planted facts, with models side by side.

**Use mode:** upload or fetch any document, give it a JSON schema, and get the extracted JSON back. Same code path, no scoring.

## Models

Support any OpenAI-compatible endpoint, configured by base URL, API key env var and model name. Default to OpenRouter with:
- `moonshotai/kimi-linear-48b-a3b-instruct` (the hybrid linear-attention model under test)
- one Claude model, as a full-attention reference

Use **Instructor** for structured output. Don't write your own JSON parsing or retry logic. In each run, record the serving provider OpenRouter reports, the context length actually sent, token counts and cost. Read API keys from environment variables only. Never write them to files, logs or run output.

## Output

Save each run as one JSON file in `eval/runs/`, and commit the runs, because the post's figures are drawn from them. The page lists runs and draws the heatmaps from them. Add a cheap "smoke" preset: one filing, 8k only, two models, three fields each. A full grid should be one button, with a cost estimate shown before it starts.

## Tests

Small pytest tests for the parts that decide the numbers:
- number and date normalisation
- fact planting at the right depth
- truncation to a token budget
- exact-match scoring

No tests for the UI.

## Not in scope

Auth, a database, multiple users, deployment, Docker, PDF parsing, and self-hosting models. Serving-speed and memory measurements are out of scope too, because an API can't show them.

## Done when

- `uv run` starts the app and the smoke preset completes against OpenRouter.
- `uv run pytest` passes.
- Nothing outside `eval/` has changed, apart from `.gitignore`.
- `eval/README.md` explains how to run it, which env vars it needs, and what the heatmaps mean.
- The README includes this caveat: "Kimi Linear vs Claude differs in size and training, not just attention, so gaps here are not a clean test of the architecture."
