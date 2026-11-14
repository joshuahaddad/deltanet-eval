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

- **Run smoke**: one filing (Apple FY2024), 8k tokens, every model, three fields (fiscal year end, one planted code, one planted note). It costs a few cents.
- **Run full grid…**: seven FY2024 filings, 45k to 208k tokens long (AAPL, MSFT, GE, KO, F, PFE and BAC), at 8k, 16k, 32k, 64k and 128k tokens and full length, every field. It shows a cost estimate before it starts, from OpenRouter's prices where a model has one. Hugging Face and Claude Code models are listed as unpriced. Each model's part of a full grid is about 2.5M input tokens: about $0.25 per Qwen model at DeepInfra's rates (Qwen3-Next is $0.09 per million input tokens and $1.10 per million output tokens; Qwen3-235B is $0.09 and $0.55).
- **Use mode**: fetch a 10-K by ticker and fiscal year, or upload an `.htm`, `.html` or `.txt` file, and give a JSON schema (objects, arrays, strings, numbers, booleans). It uses the same extraction code as the eval, with no scoring.

## Models

`models.json` lists the models under test, the judge and, under `judge_check`, an optional second judge (see [Judge check](#judge-check)). Every model listed runs in every run. Each entry is one of two kinds:

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
- **One child run per model** holds the heatmap cells as metrics: `acc/<kind>_<source>` overall and `acc/<kind>_<source>/<length>`, with matching `n/...` counts and Wilson 95% bounds `acc_lo/...` and `acc_hi/...`. Exact outcome rates are `outcome/exact_<source>/<correct|near_miss|wrong|abstained>`, and the hallucination rate is `halluc/exact_real` over `n/absent_exact_real` facts. It also has `calls_failed`, `prompt_tokens`, `cost_usd` and `notional_cost_usd`, plus two artifacts:
  - `answers.csv`: one row per question, with what the model returned, the truth, the depth, how it was scored and, for exact fields, its `outcome`.
  - `calls.json`: provider, tokens, cost and any error for each call.

- **A judge check**, when you run one, adds `judge_check/agreement`, `judge_check/kappa` and `judge_check/n` to the parent run, plus `judge_check.json`.

The run file in `runs/` stays the record. If logging to MLflow fails, the run still finishes, and the error shows next to it on the page and as `mlflow_error` in the file. To log an older run, run:

```bash
uv run python -c "import runs, json; print(runs.log_mlflow(json.load(open('runs/<id>.json'))))"
```

## What is measured

**Documents.** The 10-K's primary HTML document from EDGAR is converted to text, with each table row on one line, cells joined by ` | `. PDFs are never used.

**Exact fields, real.** Revenue, net income, operating income, diluted EPS, total assets, total liabilities, cash, cover-page shares outstanding and the fiscal year end date. Ground truth is the XBRL fact in that filing for the stated fiscal year, and each question names the year. A field is dropped when XBRL has no single value for it, for example when a company has several share classes. Scoring parses both sides into one number: it handles `$`, commas, parentheses for negatives and the scale (`million`, `bn`, `(in millions)`, …). It requires them to be equal. A rounded answer such as "$391.0 billion" for $391,035 million counts as wrong, because exact recall is what's being tested. Dates are parsed from common formats and compared as dates.

**Tolerant fields, real.** Three questions, each with one answer that sits in a short passage:

| Field | Question | Anchor | Found in (full preset) |
|---|---|---|---|
| `employees` | About how many people the company employed at year end | the human capital headcount sentence ("approximately 164,000 full-time equivalent employees"), or Ford's employment table | all 7 |
| `seasonality` | When in the year sales or production peak, and why | a `Seasonality` heading | AAPL, MSFT, KO, F |
| `revenue_drivers` | What drove the change in total revenue | an MD&A sentence on *total* revenue ("total revenues increased…", "Net operating revenues were…") | MSFT, GE, KO, PFE |

The passage runs from the anchor's line to the end of the line about 1,500 characters on. A filing with no anchor match drops the question; segment-level sentences ("Americas net sales increased…") are deliberately not anchors, since they answer a different question. The judge grades each answer against **only the part of the passage that was in the context sent**. If less than half of the passage was sent, the question counts as not present and isn't scored, like a real exact fact that was cut off. Earlier versions asked for "the two main risk factors", which has no single right answer, and graded against the first 12,000 characters of whole sections, which often missed the answer.

**Planted facts.** Near 10, 30, 50, 70 and 90% of the context that is sent, the app inserts three paragraphs, 2% apart, each with a made-up reference code (exact, for example `QX-48213-7`) and a short made-up statement (tolerant). That gives 15 exact and 15 tolerant planted questions per call, and about 21 answers per planted heatmap cell across the full grid. Values are seeded per filing, so every model and length sees the same ones.

**Outcomes of exact answers.** Strict accuracy is the headline, but every exact answer also gets one outcome:

- `correct`: an exact match, as above.
- `near_miss`: close but not exact. For numbers, the answer is the truth rounded to the precision it was given in ("$391.0 billion" or "$391 billion" for $391,035 million, "$6.1" for $6.08), or within 1%. Zero or the wrong sign is never a near miss, and nor is a missing scale ("391,035" for $391,035 million). For dates, within 7 days or the right month and year: 52/53-week fiscal years end on a weekday near the month end, so "September 30" for September 28 is a near miss, while the wrong year is wrong. For planted codes, one or two character edits away.
- `abstained`: null, empty, or "null", "none" or "n/a".
- `wrong`: anything else.

The page shows the breakdown per model, separately for real and planted facts.

**Hallucination.** Real exact facts that are not in the context aren't scored for accuracy, but they are still asked. The hallucination rate is the share of those the model answered anyway instead of returning null. It's an upper bound: a value printed in a form `locate` doesn't recognise counts as absent.

**Length.** The filing is cut to 8k, 32k and 128k tokens, and also sent whole when it fits the model. A length is skipped when the filing is shorter than it. The budget covers the whole prompt: 2,048 tokens are kept free for instructions, schema and answer. Tokens are counted with `o200k_base`. Each model's own tokenizer differs, which is why the reported prompt tokens are recorded too.

## Judge check

The judge (Claude Haiku through Claude Code) is an LLM and from the same family as one contestant, so its verdicts need checking. **Check judge** on a finished run re-grades a fixed sample of up to 60 tolerant answers, seeded by the run ID, with the `judge_check` model in `models.json` (Claude Sonnet 5.5 through Claude Code by default; any entry kind works). Only answers the first judge actually graded are sampled; blank answers are marked wrong without a judge. It never runs on its own, because it uses quota. The result goes into the run file as `judge_check`: raw agreement, Cohen's κ (agreement corrected for chance; undefined when both judges give one verdict to everything), each item's two verdicts, and the cost. It is logged to MLflow when the run was. Agreement doesn't show either judge is right, only how much the score depends on the choice of judge. Sonnet is also a contestant, so swap in a model from another family for a stricter check.

## Reading the heatmaps

There are four panels: exact or tolerant, crossed with real or planted, with the models side by side in each. Rows are depth, binned to 10/30/50/70/90%. Columns are context length. A cell shows the percentage correct and the count, and a `·` means no data. Hover a cell for its Wilson 95% interval: with about 20 answers per cell, a 70% cell is roughly 48–85%, so differences of a few cells' worth are noise.

- **Depth** is where the fact sits in the text that was sent, as a fraction of its length. For planted facts, that is where they were inserted. For a real exact field, it is the first place the value is printed. For a real tolerant field, it is where its passage starts.
- Real facts only count when they are inside the context that was sent. At 8k, most financial figures are cut off. Real facts also cluster: cover-page items near 0%, the financial statements around 60–80%. **Use the planted panels for clean depth curves** and the real panels to check that the effect holds on real content.
- The post's claim predicts this: the exact panels fade for the hybrid as length grows or as facts sit deeper, while the tolerant panels stay flat for both models.

## Files

- `app.py`: the web API and `main()`.
- `edgar.py`: EDGAR fetching and caching, HTML to text, passage finding, the exact and tolerant questions and ground truth.
- `scoring.py`: number and date parsing, exact match, outcomes, locating facts, Wilson intervals and judge agreement.
- `context.py`: planted facts, truncation and token counting, and building the context sent.
- `llm.py`: `models.json`, JSON schema to pydantic, model calls (Instructor and Claude Code), extraction and the judge.
- `runs.py`: presets, cost estimates, the run and judge-check lifecycle, and MLflow logging.
- `paths.py`: the shared paths.
- `index.html`: the single page.
- `models.json`: endpoints and models.
- `runs/`: one JSON file per run, committed because the post's figures are drawn from them.
- `mlflow.db`, `mlruns/`: the local MLflow store (git-ignored).
- `tests/`: normalisation, planting depth, truncation, exact-match and outcome scoring, passage finding and clipping, Wilson intervals, the metrics and the judge-check agreement.
- `.cache/`: EDGAR responses and converted text (git-ignored).

Not in scope: auth, a database, deployment, PDF parsing, self-hosted models, and speed or memory measurements, which an API can't show.
