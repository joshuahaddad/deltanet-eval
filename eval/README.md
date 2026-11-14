# filing-recall

A small long-context extraction eval on SEC 10-K filings. It measures *exact* recall (figures, dates, IDs) against *tolerant* recall (gist), by depth in the document and by context length. It compares a linear-attention hybrid (3 delta-rule layers to 1 full-attention layer) with full-attention models. It is also a plain extraction tool: give it a document and a JSON schema, and it returns JSON.

> Kimi Linear vs Claude differs in size and training, not just attention, so gaps here are not a clean test of the architecture.

> **Models actually run.** Kimi Linear (Kimi Delta Attention) has no working hosted endpoint as of November 2026, and Gated DeltaNet-2 has no released model larger than about 1.3B. The default line-up is therefore:
> - **Qwen3-Next-80B-A3B:** the same 3:1 hybrid design, built on Gated DeltaNet, the layer KDA refines.
> - **Qwen3-235B-A22B:** a full-attention model from the same family, as a closer control than Claude.
> - **Claude Sonnet 5.5:** as the frontier reference.
>
> The same caveat applies to every pair: Qwen3-Next and Qwen3-235B are closer in training but still differ in size. Results speak to Gated DeltaNet hybrids, not to KDA or Gated DeltaNet-2.

## Run

Needs [uv](https://docs.astral.sh/uv/). From this folder:

```bash
export HF_TOKEN=hf_...                # the Qwen models via Hugging Face (Claude runs through your local `claude` login)
export EDGAR_CONTACT=you@example.com # SEC requires a contact in the User-Agent
uv run app.py                        # http://127.0.0.1:8765 (set PORT to change)
uv run pytest
```

| Env var | Used for |
|---|---|
| `HF_TOKEN` | the Qwen models, through Hugging Face's router to DeepInfra. A fine-grained token with "Make calls to Inference Providers". |
| `OPENROUTER_API_KEY` | optional: any OpenRouter model you add to `models.json` |
| `LITELLM_API_KEY` | optional: a LiteLLM gateway entry (see Models) |
| `ANTHROPIC_API_KEY` | optional: when set, `claude -p` bills it instead of your Claude subscription |
| `EDGAR_CONTACT` | the `User-Agent: filing-recall <contact>` header that EDGAR requires |
| `PORT` | optional, default 8765 |
| `MLFLOW_TRACKING_URI` | optional: your MLflow server, e.g. `http://host:5000`. Unset logs to the local `eval/mlflow.db`. |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | optional: for a server with auth turned on (MLflow reads them itself) |
| `MLFLOW_EXPERIMENT_NAME` | optional, default `filing-recall` |

Keys are read from the environment only and are never written to runs or logs.

On the page:

- **Run smoke**: one filing (Apple FY2024), 8k tokens, both models, three fields (fiscal year end, one planted code, one planted note). It costs a few cents.
- **Run full grid…**: three filings (AAPL, MSFT and KO for FY2024) at 8k, 32k and 128k tokens and full length, every field. It shows a cost estimate before it starts, from OpenRouter's prices where a model has one. Hugging Face and Claude Code models are listed as unpriced. Each Qwen model's part of a full grid is about 370k input tokens: about $0.05 at DeepInfra's rates (Qwen3-Next is $0.09 per million input tokens and $1.10 per million output tokens; Qwen3-235B is $0.09 and $0.55).
- **Use mode**: fetch a 10-K by ticker and fiscal year, or upload an `.htm`, `.html` or `.txt` file, and give a JSON schema (objects, arrays, strings, numbers, booleans). It uses the same extraction code as the eval, with no scoring.

## Models

`models.json` lists the models under test and the judge. Every model listed runs in every run. Each entry is one of two kinds:

- **OpenAI-compatible endpoint:** `base_url`, `api_key_env` and `model`. Structured output goes through [Instructor](https://python.useinstructor.com/) in JSON mode, which handles parsing and retries. OpenRouter's catalogue supplies context length and price. For any other endpoint, add `"context_length"`.
- **`"provider": "claude-code"`:** `model` and `context_length`. The call runs the local `claude -p` with no tools, no settings, no MCP servers and `--json-schema`, so it uses your Claude Code login (your subscription). If `ANTHROPIC_API_KEY` is set, Claude Code bills that key instead. Claude Code enforces the schema itself, and pydantic validates the result. Under a subscription, the run records `"billing": "subscription"`, leaves `cost` empty and puts the list-price figure in `notional_cost`. The full grid then uses up subscription quota, not money.

By default, both Qwen models go through Hugging Face's router to DeepInfra, which serves 262k tokens of context. The `:deepinfra` suffix pins the provider, so results don't drift between hosts. To run Kimi Linear once a host serves it, add `moonshotai/Kimi-Linear-48B-A3B-Instruct` with that host's suffix (Hugging Face listed Featherless AI, which returned "at capacity" in November 2026). Claude Sonnet 5.5 and the judge (Claude Haiku 5.5) go through Claude Code. The judge is a different Claude model from the one under test, so Claude never grades its own answers.

A LiteLLM gateway, for local models, is an OpenAI-compatible entry. The `model` is the gateway's model name, and `context_length` is what the gateway reports as `max_input_tokens`:

```json
{"name": "Local coder", "base_url": "http://localhost:4000/v1", "api_key_env": "LITELLM_API_KEY",
 "model": "local-coder", "context_length": 131072}
```

Each call records the serving provider (as the endpoint reports it, or `claude-code/firstParty`), the tokens sent (counted locally and as reported), the completion tokens and the cost. Reported prompt tokens include each model's own tokenizer and harness overhead. For example, an 8k cell of about 6.1k `o200k` tokens comes to about 11.3k tokens through Claude Code.

## MLflow

Each finished run is also logged to MLflow. By default it goes to a local store, `eval/mlflow.db` with artifacts in `eval/mlruns/`, both git-ignored. Browse it with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To use your server instead, set `MLFLOW_TRACKING_URI`, plus the username and password if it uses auth.

- **The parent run** is named after the run ID. It holds the run's parameters (preset, judge, filings, status), `total_cost_usd` and the whole run file as `run.json`.
- **One child run per model** holds the heatmap cells as metrics: `acc/<kind>_<source>` overall and `acc/<kind>_<source>/<length>`, with matching `n/...` counts. It also has `calls_failed`, `prompt_tokens`, `cost_usd` and `notional_cost_usd`, plus two artifacts:
  - `answers.csv`: one row per question, with what the model returned, the truth, the depth and how it was scored.
  - `calls.json`: provider, tokens, cost and any error for each call.

The run file in `runs/` stays the record. If logging to MLflow fails, the run still finishes, and the error shows next to it on the page and as `mlflow_error` in the file. To log an older run, run:

```bash
uv run python -c "import app, json; print(app.log_mlflow(json.load(open('runs/<id>.json'))))"
```

## What is measured

**Documents.** The 10-K's primary HTML document from EDGAR is converted to text, with each table row on one line, cells joined by ` | `. PDFs are never used.

**Exact fields, real.** Revenue, net income, operating income, diluted EPS, total assets, total liabilities, cash, cover-page shares outstanding and the fiscal year end date. Ground truth is the XBRL fact in that filing for the stated fiscal year, and each question names the year. A field is dropped when XBRL has no single value for it, for example when a company has several share classes. Scoring parses both sides into one number: it handles `$`, commas, parentheses for negatives and the scale (`million`, `bn`, `(in millions)`, …). It requires them to be equal. A rounded answer such as "$391.0 billion" for $391,035 million counts as wrong, because exact recall is what's being tested. Dates are parsed from common formats and compared as dates.

**Tolerant fields, real.** The main products or services (Item 1), the two main risk factors (Item 1A) and what drove the change in revenue (Item 7). An LLM judge grades each answer against the opening of that section.

**Planted facts.** At 10, 30, 50, 70 and 90% of the context that is sent, the app inserts a paragraph with a made-up reference code (exact, for example `QX-48213-7`) and a short made-up statement (tolerant). Values are seeded per filing, so every model and length sees the same ones.

**Length.** The filing is cut to 8k, 32k and 128k tokens, and also sent whole when it fits the model. A length is skipped when the filing is shorter than it. The budget covers the whole prompt: 2,048 tokens are kept free for instructions, schema and answer. Tokens are counted with `o200k_base`. Each model's own tokenizer differs, which is why the reported prompt tokens are recorded too.

## Reading the heatmaps

There are four panels: exact or tolerant, crossed with real or planted, with the models side by side in each. Rows are depth, binned to 10/30/50/70/90%. Columns are context length. A cell shows the percentage correct and the count, and a `·` means no data.

- **Depth** is where the fact sits in the text that was sent, as a fraction of its length. For planted facts, that is where they were inserted. For a real exact field, it is the first place the value is printed. For a real tolerant field, it is where the section starts.
- Real facts only count when they are inside the context that was sent. At 8k, most financial figures are cut off. Real facts also cluster: cover-page items near 0%, the financial statements around 60–80%. **Use the planted panels for clean depth curves** and the real panels to check that the effect holds on real content.
- The post's claim predicts this: the exact panels fade for the hybrid as length grows or as facts sit deeper, while the tolerant panels stay flat for both models.

## Files

- `app.py`: everything, including EDGAR fetching, ground truth, planting, truncation, extraction, scoring, runs and the web API.
- `index.html`: the single page.
- `models.json`: endpoints and models.
- `runs/`: one JSON file per run, committed because the post's figures are drawn from them.
- `mlflow.db`, `mlruns/`: the local MLflow store (git-ignored).
- `tests/`: normalisation, planting depth, truncation and exact-match scoring.
- `.cache/`: EDGAR responses and converted text (git-ignored).

Not in scope: auth, a database, deployment, PDF parsing, self-hosted models, and speed or memory measurements, which an API can't show.
