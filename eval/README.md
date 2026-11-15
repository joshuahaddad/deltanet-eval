# filing-recall

A small long-context extraction eval on SEC 10-K filings. It measures *exact* recall (figures, dates, IDs) against *tolerant* recall (gist), by depth in the document and by context length. It compares a linear-attention hybrid (3 delta-rule layers to 1 full-attention layer) with full-attention models. It is also a plain extraction tool: give it a document and a JSON schema, and it returns JSON.

A second mode, [HELMET ICL](#helmet-icl-mode), covers the other half of the claim: many-shot in-context learning on a banking intent-classification dataset, a long-context task that rewards the gist of thousands of examples rather than one exact string.

> The models compared differ in size and training, not just attention, so gaps here are not a clean test of the architecture.

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
uv run app.py                        # http://127.0.0.1:8780 (set PORT to change)
uv run pytest
```

| Env var | Used for |
|---|---|
| `HF_TOKEN` | the Qwen models, through Hugging Face's router to DeepInfra. A fine-grained token with "Make calls to Inference Providers". |
| `OPENROUTER_API_KEY` | optional: any OpenRouter model you add to `models.json` |
| `LITELLM_API_KEY` | optional: a LiteLLM gateway entry (see Models) |
| `ANTHROPIC_API_KEY` | optional: when set, `claude -p` bills it instead of your Claude subscription |
| `EDGAR_CONTACT` | the `User-Agent: filing-recall <contact>` header that EDGAR requires |
| `PORT` | optional, default 8780 (8765 is AnkiConnect's port) |
| `MLFLOW_TRACKING_URI` | optional: your MLflow server, e.g. `http://host:5000`. Unset logs to the local `eval/mlflow.db`. |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | optional: for a server with auth turned on (MLflow reads them itself) |
| `MLFLOW_EXPERIMENT_NAME` | optional, default `filing-recall` |

Keys are read from the environment only and are never written to runs or logs.

On the page:

- **Run smoke**: one filing (Apple FY2024), 8k tokens, every model, four fields (fiscal year end, revenue, one planted code, one planted note), in every call mode. Single calls ask planted codes only, so that is 3 calls per model, 9 in all, and under a cent.
- **Run full grid…**: ten FY2024 filings (AAPL, MSFT, GE, KO, F, PFE, BAC, JPM, C and WFC), 21k to 299k tokens long, at 8k, 32k, 128k and 192k tokens, every field, in every call mode (see [Call modes](#what-is-measured)). 192k stays inside Qwen's 262k window, with a margin for Qwen's own tokenizer, which differs from `o200k_base`. Which filings reach which lengths:

  | Length | Filings | Count |
  |---|---|---|
  | 8k | all ten | 10 |
  | 32k | all but WFC | 9 |
  | 128k | F, PFE, BAC, JPM, C (KO, at 126k tokens, is just short) | 5 |
  | 192k | BAC, JPM, C | 3 |

  WFC's primary 10-K document is only about 21k tokens, because its annual report is filed as an exhibit, so it contributes at 8k only. Claude runs at 32k only, because it uses subscription quota. Each filing, length and model gets 1 batched call and 15 single calls, and each filing and model gets one closed-book call. That is 1,479 calls: 442 per Qwen model and 153 for Claude. **Run full grid** runs every model in `models.json`; the `full-30b` preset (`POST /api/runs/full-30b`) runs the same grid for Qwen3-30B-A3B only, whose comparison models are in `runs/20261114-193139-full.json`. Its estimate: about 27.8M input tokens, about $3.49 at Featherless's rates, from the tokens Qwen3-Next reported in that run (the same tokenizer).
- The page shows a cost estimate before a grid starts, from the prices OpenRouter or the Hugging Face router (`https://router.huggingface.co/v1/models`) list for the pinned provider. Claude Code models, Claude and the judge, are listed as unpriced. For the full grid, each Qwen model gets about 25.2M input tokens: about $2.36 for Qwen3-Next and $2.32 for Qwen3-235B, $4.68 together, at DeepInfra's rates (both $0.09 per million input tokens; output is $1.10 and $0.55 per million). By length, the two together cost about $0.24 at 8k, $0.85 at 32k, $1.89 at 128k and $1.70 at 192k. Claude gets about 4.47M input tokens, on quota. Most of the tokens are the single calls, which each send the whole context for one question. The first full run cost about 4% more than its estimate (see [First run](#first-run)). The estimate also returns `usd_per_model` and `missing_filings`: a filing that can't be loaded is listed there and left out of the count instead of failing the estimate.
- **Retry failed calls**: on a finished filing run, the page's retry button (or `POST /api/runs/{rid}/retry`) re-runs only the calls that errored and replaces their results in place. It then recomputes the paired stats and the total cost and logs a new MLflow run; the old one stays, named in the run file's `retry.previous_mlflow_run_id`. The `retry` key also records the status, the number of calls retried and how many still failed. Contexts are rebuilt from the code, so a retry is refused when the run's `design` differs from the code's. Runs from before `design` existed have none and can't be retried. Only one job (the run, a retry or a judge check) may write a run file at a time.
- **Run ICL smoke** and **Run ICL grid…**: HELMET's many-shot ICL task; see [HELMET ICL mode](#helmet-icl-mode). The grid shows its call count and input tokens per model before it starts.
- **Use mode**: fetch a 10-K by ticker and fiscal year, or upload an `.htm`, `.html` or `.txt` file, and give a JSON schema (objects, arrays, strings, numbers, booleans). It uses the same extraction code as the eval, with no scoring.

## Models

`models.json` lists the models under test, the judge and, under `judge_check`, an optional second judge (see [Judge check](#judge-check)). Every model listed runs in every run. Each entry is one of two kinds:

- **OpenAI-compatible endpoint:** `base_url`, `api_key_env` and `model`. Structured output goes through [Instructor](https://python.useinstructor.com/) in JSON mode, which handles parsing and retries. OpenRouter's catalogue, or the Hugging Face router's listing for the provider after the `:` suffix, supplies context length and price. The Hugging Face router reports tokens but no cost, so the cost is computed from that listed price. For any other endpoint, add `"context_length"`. When the router doesn't list the model's provider, or lists it with no price, add `"context_length"` and `"pricing": {"input": ..., "output": ...}` in dollars per million tokens, from the provider's own page.
- **`"provider": "claude-code"`:** `model` and `context_length`. The call runs the local `claude -p` with no tools, no settings, no MCP servers and `--json-schema`, so it uses your Claude Code login (your subscription). If `ANTHROPIC_API_KEY` is set, Claude Code bills that key instead. Claude Code enforces the schema itself, and pydantic validates the result. Under a subscription, the run records `"billing": "subscription"`, leaves `cost` empty and puts the list-price figure in `notional_cost`. The full grid then uses up subscription quota, not money.

By default, both Qwen models go through Hugging Face's router to DeepInfra, which serves 262k tokens of context. The `:deepinfra` suffix pins the provider, so results don't drift between hosts. To run Kimi Linear once a host serves it, add `moonshotai/Kimi-Linear-48B-A3B-Instruct` with that host's suffix (Hugging Face listed Featherless AI, which returned "at capacity" in November 2026). Claude Sonnet 5.5 and the judge (Claude Haiku 5.5) go through Claude Code. The judge is a different Claude model from the one under test, so Claude never grades its own answers. The judge check (DeepSeek-V4-Flash) goes through the Hugging Face router to DeepInfra.

Qwen3-30B-A3B-Instruct-2507 is the full-attention control at the hybrid's scale: 3B active parameters like Qwen3-Next, where Qwen3-235B has 22B. The Hugging Face router serves it only through Featherless AI (`:featherless-ai`), which isn't in the router's public listing, so its entry sets `context_length` (262,144) and `pricing` ($0.125 in, $0.45 out per million tokens, FP8, from [Featherless's page](https://featherless.ai/models/Qwen/Qwen3-30B-A3B-Instruct-2507), November 2026). Its provider and weight format differ from DeepInfra's for the other two, so a gap against Qwen3-Next mixes attention with serving.

A LiteLLM gateway, for local models, is an OpenAI-compatible entry. The `model` is the gateway's model name, and `context_length` is what the gateway reports as `max_input_tokens`:

```json
{"name": "Local coder", "base_url": "http://localhost:4000/v1", "api_key_env": "LITELLM_API_KEY",
 "model": "local-coder", "context_length": 131072}
```

Each call records the serving provider (as the endpoint reports it, or `claude-code/firstParty`), the tokens sent (counted locally and as reported), the completion tokens, the cost, the wall-clock latency (`latency_s`) and, for OpenAI-compatible endpoints, the number of `attempts`. A 429, 500, 502, 503 or 504, or a timeout, gets up to 3 more attempts, after waits of 10, 30 and 60 seconds, each stretched by up to 20% of random jitter. Other errors fail at once. `latency_s` is the answering attempt only, without the waits. Instructor's own retries, up to 2, cover answers that fail validation, not transport errors. Reported prompt tokens include each model's own tokenizer and harness overhead. For example, an 8k cell of about 6.1k `o200k` tokens comes to about 11.3k tokens through Claude Code.

## MLflow

Each finished run is also logged to MLflow. By default it goes to a local store, `eval/mlflow.db` with artifacts in `eval/mlruns/`, both git-ignored. Browse it with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To use your server instead, set `MLFLOW_TRACKING_URI`, plus the username and password if it uses auth.

- **The parent run** is named after the run ID. It holds the run's parameters (preset, judge, filings, status), `total_cost_usd` and the whole run file as `run.json`. It also holds the paired headline (see [Reading the results](#reading-the-results)) as `paired/<A vs B>/<mode>/<stat>/<length>`, with bounds `paired_lo/...` and `paired_hi/...`. `<mode>` is `batched` or `single`, `<stat>` is `exact_gap`, `tolerant_gap` or `did` (the last two in batched only), and `<length>` is `8k`, `32k`, `128k`, `192k` or `all`. Paired entries from run files before call modes have no mode and are logged as `single`, which is what they were. The pair name drops characters MLflow doesn't allow, so "Qwen3 235B-A22B (full attention)" becomes "Qwen3 235B-A22B full attention".
- **One child run per model** holds the cells as metrics, keyed by call mode (`batched`, `single` or `closed_book`):
  - `acc/<mode>/<kind>_<source>` overall, `acc/<mode>/<kind>_<source>/<length>` and `acc/<mode>/<kind>_<source>/d<10|30|50|70|90>` per depth bin, pooled over lengths, with matching `n/...` counts and Wilson 95% bounds `acc_lo/...` and `acc_hi/...`. Closed-book calls have no length or depth, so they log only `acc/closed_book/exact_real`.
  - `acc/batched/exact_real_unmemorised`: batched real exact accuracy without the facts the model got right closed-book. Logged only when the run has closed-book calls.
  - Exact outcome rates: `outcome/<mode>/exact_<source>/<correct|decoy|near_miss|wrong|abstained>`.
  - The hallucination rate, `halluc/exact_real`, over `n/absent_exact_real` facts, from batched calls.
  - Per mode: `calls_failed/<mode>`, `notional_cost_usd/<mode>`, `cost_per_correct/<mode>` and `latency_s_median/<mode>`. Totals across modes: `calls`, `calls_failed`, `prompt_tokens`, `cost_usd` and `notional_cost_usd`.
  - Batched cells also keep their old names, without the mode: `acc/<kind>_<source>`, `acc/<kind>_<source>/<length>` (with their `n`, `acc_lo` and `acc_hi`) and `outcome/exact_<source>/<outcome>`. Runs from before call modes count as batched, so old and new runs line up on those names.
  - Two artifacts:
    - `answers.csv`: one row per question, with the call mode, what the model returned, the truth, the decoy codes for planted codes (`decoy` in old runs, `decoys` in new ones), the depth, how it was scored and, for exact fields, its `outcome` and, for a decoy answer, `decoy_hit` (`near` or `far`).
    - `calls.json`: mode, provider, tokens, cost, latency and any error for each call.

- **ICL runs** use the same layout. The parent run's parameters are the preset, `mode=icl`, the status and the datasets. Each model's child run holds `icl_acc/<dataset>/<length>` (for example `icl_acc/banking77/32k`) with Wilson 95% bounds `icl_acc_lo/...` and `icl_acc_hi/...` and the count `icl_n/...`, the same call, token and cost metrics as above, `calls.json`, and an `answers.csv` with one row per test item: dataset, length, item, text, gold label, predicted label, correct, shots and any call error. Filing runs log exactly what they did before.
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
| `employees` | About how many people the company employed at year end | the human capital headcount sentence ("approximately 164,000 full-time equivalent employees"), or Ford's employment table | all but JPM |
| `seasonality` | When in the year sales or production peak, and why | a `Seasonality` heading | AAPL, MSFT, KO, F |
| `revenue_drivers` | What drove the change in total revenue | an MD&A sentence on *total* revenue ("total revenues increased…", "Net operating revenues were…") | MSFT, GE, KO, PFE |

The three banks added to the full preset have no `seasonality` or `revenue_drivers` passage. Their real exact fields are also thinner: JPM has no employees, cash or operating income field, C and BAC no cash or operating income, and WFC no revenue, operating income or cash, because the app finds no single XBRL value for them.

The passage runs from the anchor's line to the end of the line about 1,500 characters on. A filing with no anchor match drops the question; segment-level sentences ("Americas net sales increased…") are deliberately not anchors, since they answer a different question. The LLM judge grades only these real tolerant answers, each against **only the part of the passage that was in the context sent**. If less than half of the passage was sent, the question counts as not present and isn't scored, like a real exact fact that was cut off. Earlier versions asked for "the two main risk factors", which has no single right answer, and graded against the first 12,000 characters of whole sections, which often missed the answer.

**Planted facts.** Near 10, 30, 50, 70 and 90% of the context that is sent, the app inserts three paragraphs, 2% apart, each with a made-up reference code (exact, for example `QX-48213-7`) and a short made-up statement (tolerant). That gives 15 exact and 15 tolerant planted questions per filing and length. Values are seeded per filing, so every model, length and mode sees the same ones.

**Decoys.** Each planted code has two near-duplicates that are never asked about: the same kind of paragraph for a lookalike subject, with a different code and a filler sentence.

- **Near:** 0.5% after its target, in the same depth bin ("the Brookfield archive survey" against the asked "Brookfield archive audit"). It sits halfway between the target and the next 2%-spaced plant or far decoy, so it never shares their line.
- **Far:** 40 points deeper, wrapped into the 10–90% bins (10 to 50, 30 to 70, 50 to 90, 70 to 10, 90 to 30), plus 1% ("the Brookfield archive review"), so it falls between the 2%-spaced plants.

Each kind of decoy code comes from its own seeded stream, so the real codes and the far decoy codes are the same as in design 1, before near decoys existed. An answer that gives either decoy's code is the `decoy` outcome below: the model found a paragraph, but the wrong one. `decoy_hit` says which one (`near` or `far`). Planted code questions carry `decoys`, `decoy_targets` and `decoy_depths`, each keyed `near` and `far`; design 1 runs have a single `decoy` and `decoy_depth` instead. `DESIGN` in `context.py` is 2, and every new run file records it as `design`.

**Planted tolerant answers are scored by stems, not by the judge.** Each planted statement has one or two stems, and a correct answer must contain every one as a lowercase substring. `a|b` accepts either: for "drivers reported fatigue", the stems are `driver` and `fatigu|tired|exhaust`. This is cheap, deterministic and needs no judge, but it can miss a correct paraphrase (see [Design choices and limits](#design-choices-and-limits)).

**Call modes.** Each filing and length is asked in more than one way, all with the same context:

- `batched`: every question in one call, as a practitioner would extract a filing. Real and planted questions are asked together.
- `single`: one call per planted code, 15 per filing and length, so no question can interfere with another. Planted notes and real questions are only batched. Notes were at the ceiling in single mode in the [first run](#first-run) (96–100%), so single calls for them spent money without telling the models apart. Real questions would multiply the cost.
- `closed_book`: the real questions with an empty document, once per filing, to measure what the model knows from memory. Its instruction differs from the open-book one: it says the filing is not shown, asks for an answer from memory, and the request names the company and fiscal year ("The filing is AAPL's 10-K for fiscal year 2024"). The open-book instruction says to answer from the document only, which would force null. Closed-book tolerant answers are recorded but not judged.

The gap between `batched` and `single` on the same planted codes is itself a measurement: what batching costs in accuracy.

**Outcomes of exact answers.** Strict accuracy is the headline, but every exact answer also gets one outcome:

- `correct`: an exact match, as above.
- `decoy`: for planted codes, an exact match to the code of its unasked lookalike.
- `near_miss`: close but not exact. For numbers, the answer is the truth rounded to the precision it was given in ("$391.0 billion" or "$391 billion" for $391,035 million, "$6.1" for $6.08), or within 1%. Zero or the wrong sign is never a near miss, and nor is a missing scale ("391,035" for $391,035 million). For dates, within 7 days or the right month and year: 52/53-week fiscal years end on a weekday near the month end, so "September 30" for September 28 is a near miss, while the wrong year is wrong. For planted codes, one or two character edits away.
- `abstained`: null, empty, or "null", "none" or "n/a".
- `wrong`: anything else.

The page shows the breakdown per model, separately for real and planted facts.

**Hallucination.** Real exact facts that are not in the context aren't scored for accuracy, but they are still asked. The hallucination rate is the share of those the model answered anyway instead of returning null, from batched calls. It's an upper bound: a value printed in a form `locate` doesn't recognise counts as absent, and a model may answer correctly from memory, which closed-book calls measure.

**Length.** The full preset cuts the filing to 8k, 32k, 128k and 192k tokens. A length is skipped when the filing is shorter than it, so 32k has nine filings, 128k five and 192k three (see the table under [Run](#run)). A preset can also list `"full"`, the whole filing when it fits the model, but none does now. The budget covers the whole prompt: 2,048 tokens are kept free for instructions, schema and answer. Tokens are counted with `o200k_base`. Each model's own tokenizer differs, which is why the reported prompt tokens are recorded too.

## Judge check

The judge (Claude Haiku through Claude Code) is an LLM and from the same family as one contestant, so its verdicts need checking. **Check judge** on a finished run re-grades a fixed sample of up to 60 real tolerant answers, seeded by the run ID, with the `judge_check` model in `models.json`. Planted tolerant answers are scored by stems, so they never reach a judge and aren't sampled. Only answers the first judge actually graded are sampled; blank answers are marked wrong without a judge. It never runs on its own, because it costs money. The result goes into the run file as `judge_check`: raw agreement, Cohen's κ (agreement corrected for chance; undefined when both judges give one verdict to everything), each item's two verdicts, and the cost. It is logged to MLflow when the run was. Agreement doesn't show either judge is right, only how much the score depends on the choice of judge.

The default check model is DeepSeek-V4-Flash, through the Hugging Face router to DeepInfra. It is from a different family from every contestant and from the first judge, so no model checks its own family's answers. DeepInfra lists it at about $0.09 per million input tokens and $0.18 per million output tokens, so 60 short grading prompts cost well under a cent. Any entry kind works in `judge_check`.

## Reading the results

**Heatmaps.** There are four panels: exact or tolerant, crossed with real or planted, with the models side by side in each. Rows are depth, binned to 10/30/50/70/90%. Columns are context length. A cell shows the percentage correct and the count, and a `·` means no data. Hover a cell for its Wilson 95% interval. A planted cell holds 3 questions per filing, per model and mode: 30 at 8k, 27 at 32k, 15 at 128k and 9 at 192k. Planted notes are batched only, so the tolerant planted panel has no single-mode cells. A 70% cell is roughly 52–83% at 30 answers and 35–88% at 9, so differences of a few cells' worth are noise.

**Pooled depth curves.** Accuracy per depth bin, pooled over lengths (the `d10` to `d90` metrics). Pooling gives each Qwen model 81 planted questions per bin and mode (30 + 27 + 15 + 9), with a 70% point at roughly 60–79%. Claude has 27 per bin, all at 32k. The pooled curve mixes lengths, and the long lengths contribute fewer items, so read it for depth, and the heatmap columns for length.

**Paired headline.** For each call mode and pair of models, on the planted questions both answered, the run file's `paired` list holds up to three gaps, per length and pooled (`all`), each with a 95% bootstrap interval. Each entry has a `mode`. Single entries have only `exact_gap`, since single calls ask codes only; `tolerant_gap` and `did` are null there. Batched entries have all three:

- `exact_gap`: model A's accuracy minus model B's on planted codes.
- `tolerant_gap`: the same on planted notes.
- `did`: the difference in differences, exact gap minus tolerant gap, pairing each code with the note from the same paragraph, in the same batched call. In design 1 it was measured in single mode. The post's claim predicts it is negative when A is the hybrid and B a full-attention model: the hybrid loses more on exact recall than on gist. A is whichever model comes first in `models.json`, so with the default line-up the hybrid is A.

Pairing is by ticker, fiscal year, length and field, so each item is a like-for-like comparison. Each Qwen pair has 15 codes per filing and length: 150 at 8k, 135 at 32k, 75 at 128k and 45 at 192k (405 pooled), in each mode. In batched mode the same counts hold for notes and for `did` pairs. A pair with Claude has only the 135 at 32k. The bootstrap resamples items, not filings; see the limits below.

**Engineering table.** Per model and mode, the numbers a practitioner would act on:

- **Batching cost:** batched minus single accuracy on the same planted questions.
- **Decoy rate:** the share of planted code answers that give the lookalike's code.
- **Hallucination:** as above, on real exact facts not in the context.
- **Memorisation:** closed-book accuracy on real exact facts, and batched real exact accuracy without the facts the model knew closed-book (`exact_real_unmemorised`).
- **Latency:** the median `latency_s` per call, per mode.
- **Cost per correct answer:** the mode's list-price spend (Claude's notional cost included) over its correct answers. It leaves out the judge's cost.

- **Depth** is where the fact sits in the text that was sent, as a fraction of its length. For planted facts, that is where they were inserted. For a real exact field, it is the first place the value is printed. For a real tolerant field, it is where its passage starts.
- Real facts only count when they are inside the context that was sent. At 8k, most financial figures are cut off. Real facts also cluster: cover-page items near 0%, the financial statements around 60–80%. **Use the planted panels for clean depth curves** and the real panels to check that the effect holds on real content.
- The post's claim predicts this: the exact panels fade for the hybrid as length grows or as facts sit deeper, while the tolerant panels stay flat for both models.

## First run

The first full grid (run file not kept) was in design 1: seven filings, 8k to 128k, one far decoy per code, single calls for codes and notes, no backoff. 1,292 calls. What it showed:

- **Tolerant recall was at the ceiling.** Planted notes scored 96–100% for every model, length and mode, and judged real tolerant answers 100%. They couldn't separate the models.
- **Almost every exact miss was a decoy.** Of 56 wrong planted code answers, 48 gave the far decoy's code. The rest were 7 abstentions and 1 near miss, all from Qwen3-235B in single mode.
- **The hybrid fell at 128k in batched mode,** to 73% (33 of 45) against 100% for Qwen3-235B. Only F and PFE fell, 9 of 15 each, and every miss was a decoy; BAC stayed at 15 of 15. That is two filings, not a length curve.
- **Latency.** Median batched latency was 194.5 s for Qwen3-235B, 28.5 s for the hybrid and 9.6 s for Claude. It includes provider queueing, so it is not an architecture measurement.
- **Memory.** Closed-book, Claude answered 71% of real exact facts (40 of 56), against 11% for Qwen3-235B and 5% for the hybrid.
- **Failures.** 8 calls failed: three on 429, three on 504 and two on 503.
- **Cost.** $3.91, about 4% above the $3.76 estimate.

These motivated design 2: near decoys, to make the exact task harder where it was nearly at the ceiling; single calls for codes only, with the difference in differences moved to batched mode; JPM, C and the 192k length, so the long end rests on more than BAC; backoff on transient errors; and retrying failed calls.

## Design choices and limits

- **Don't pool design 1 and design 2 runs.** Near decoys make the exact task harder than in design 1, and design 1's single calls asked notes too, so accuracy, decoy rates and paired stats aren't comparable across designs. Compare runs with the same `design` (none means 1).
- **The bootstrap resamples items, not filings.** Questions from one filing are correlated, so the paired intervals are narrower than they should be. With 10 filings, a bootstrap over filings would be too coarse to be useful. It is worth switching once there are a few dozen filings.
- **10 filings, fewer at length.** Every per-filing effect (layout, length, how the company writes) rests on ten documents at 8k, five at 128k and three at 192k. Three of the five 128k filings, and all three at 192k, are banks, so the long end leans on one kind of filing.
- **Retries change the timing, not the context.** A retried call sends the same context, rebuilt from the code, but at a later time and maybe to a less busy provider. Its latency isn't comparable to the original calls'.
- **Stems can miss paraphrases.** A correct planted note answer that uses none of a stem's words ("the trucks' crews were worn out") counts as wrong. Stems trade that for a deterministic score with no judge in the loop.
- **The closed-book instruction differs from the open-book one.** It has to: the open-book one says to answer from the document only. So closed-book accuracy is memory under a different prompt, not the same call with the document removed.
- **Latency includes provider queueing.** It is wall-clock time per call through the router or `claude -p`, on shared hardware, so it isn't an architecture measurement. Compare modes within one model rather than models.
- **Claude runs at 32k only.** It uses subscription quota, so it has no length curve, and its paired comparisons cover one length.
- **Single mode asks codes only.** So there is no single-mode tolerant panel, and batching cost is measured on codes only.

## HELMET ICL mode

**What it measures.** [HELMET](https://github.com/princeton-nlp/HELMET) (Yen et al., ICLR 2025, [arXiv:2410.02694](https://arxiv.org/abs/2410.02694)) has a many-shot in-context learning category. The prompt holds hundreds to thousands of labelled examples (`text`, then `label: N`), followed by one unlabelled text, and the model must give its label. Labels are shown as **random integers**, not intent names, so the model can't lean on what "card_arrival" means. It has to learn the mapping from the examples in the context. This mode runs HELMET's banking dataset, [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) (77 banking intents), to keep the focus on the financial domain.

**Why it's the gist case.** No single example holds the answer, and no exact string has to be copied out. The model needs a summary of what each label's examples have in common, built up over the whole context. A fixed-size recurrent state can hold that kind of summary, so the post's claim predicts that a linear-attention hybrid keeps up with full attention here, while it falls behind on the exact panels above. The Kimi Linear paper ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692), Table 5, models trained on 1.4T tokens, at 128k context) reports HELMET-ICL at 90.0 for Kimi Linear, 88.0 for full-attention MLA and 85.5 for the Gated DeltaNet hybrid (GDN-H).

**Caveat: it's partly fuzzy recall.** Many-shot ICL classification works partly by retrieval: the model finds the examples most like the test text and copies their label. That is recall, of a tolerant kind: the match is by meaning, not by exact string, and the label it copies is a short integer that repeats across many examples. So this mode tests tolerant recall over a long context more than pure abstraction. A good score doesn't show the hybrid is good at exact recall, and a gap between this mode and the planted exact codes is what the post's claim predicts.

**Presets.**

| Preset | Datasets | Lengths | Test items | Models | Calls |
|---|---|---|---|---|---|
| `icl-smoke` | BANKING77 | 32k | 5 | every model | 15 |
| `icl` | BANKING77 | 32k, 64k, 128k | 50 | every model; Claude at 32k only | 350 |

One call is one test item at one length. A preset's `model_lengths` limits a model, by its `name` or its `provider`, to some lengths. `icl` sets `{"claude-code": [32768]}`, because Claude runs on subscription quota, and so does the filing preset `full`.

**Cost.** Input tokens are counted with `o200k_base`, plus 300 per call for the instruction, schema and wrapper. Outputs are a few tokens each.

| Preset | Qwen3-Next | Qwen3-235B | Claude Sonnet (quota) |
|---|---|---|---|
| `icl-smoke` | 0.13M tokens, about $0.01 | 0.13M, about $0.01 | 0.13M |
| `icl` | 9.3M tokens, about $0.84 | 9.3M, about $0.84 | 1.32M |

The dollar figures use DeepInfra's $0.09 per million input tokens. The page can't price Hugging Face or Claude Code models, so it shows these token counts with the estimate instead. Each test item gets its own demos and label mapping, so prompt caching doesn't help. Qwen's own tokenizer and Claude Code's harness add to the reported prompt tokens. With 50 items, a 70% cell has a Wilson 95% interval of roughly 56–81%.

**How HELMET builds the prompts, and what is replicated.** From HELMET's `data.py` (`load_icl`), `configs/icl.yaml`, `configs/icl_short.yaml` and `scripts/generate_configs.py`, at commit [`aeadacc`](https://github.com/princeton-nlp/HELMET/tree/aeadacc6593e9a222a84b0660b186a4e23d2efda):

- **Data.** HELMET loads the original Hugging Face datasets, not its own `princeton-nlp/HELMET` archive. That archive holds the other tasks' data, and its card states no licence. BANKING77: `PolyAI/banking77`, demos from `train` (10,003 rows) and test items from `test` (3,080). `icl.py` downloads only those two parquet files, at pinned revisions, into `.cache/icl/`. The BANKING77 row order matches PolyAI's original `train.csv` and `test.csv`, which matters, because HELMET's sampling depends on it. **Replicated.**
- **Shots per target length.** BANKING77: 360, 720, 1,450, 2,900 and 5,900 at 8k, 16k, 32k, 64k and 128k. It is `banking77_balance`. **Replicated**, with the budget deviation below.
- **Demo selection and order.** For each test item, `local_seed = (int(sha256(text)) + 42) mod 2^31`. `balance_labels(train, shots, local_seed)` makes ceil(shots / labels) rounds holding one example per label each, shuffles each round, and cuts the list to `shots`. A label with too few examples is drawn again with replacement. **Replicated** line for line, using Python's `random` the same way.
- **Label remapping.** For each test item, `range(num_labels)` is shuffled with `random.seed(local_seed)`. Each original label shows as that integer in the demos and in the answer, so the mapping is a bijection and changes from item to item. **Replicated.**
- **Test set.** `max_test_samples` is 500. HELMET shuffles the test split with `Dataset.shuffle(seed=42)` (`numpy.random.default_rng(42).permutation`) and then balances it with `balance_labels(test, 500, 42)`. **Replicated**. Our items are the **first 50 (or 5) of HELMET's 500**, so they are a subset of its test set, and each covers a different label. Item IDs are row indices in the split. Demos come from the train split, so no test item is among the demos, and none of the 50 test texts appears verbatim in the train split.
- **Prompt template.** HELMET sends `Use the provided mapping from the text to label to assign a label to the text. Only output "label: {label}" and nothing else.`, then the demos joined by blank lines, each `<text>\nlabel: <int>`, then the test text. Generation is greedy, at most 20 tokens, and stops at a newline.
- **Parsing and scoring.** HELMET takes the text after `label:` on the first line (`parse_output`). Its reported ICL metric is `exact_match` after SQuAD-style normalisation (lowercase, no punctuation or articles), one score per dataset.

**Deviations.**

- **Structured output, not free text.** As the app requires, answers go through `llm.extract`: Instructor in JSON mode for OpenAI-compatible endpoints, `claude -p --json-schema` for Claude. The schema is a single integer field, `label`. HELMET's output instruction ("Only output label: …") is dropped, because the schema takes its place. The instruction goes in the system message. The demos go inside `extract`'s `<document>` tags, and the test text follows them. A JSON object is a different output task from continuing `label:`, and Instructor retries a malformed answer up to twice. **Absolute numbers aren't directly comparable to HELMET's or to the Kimi Linear paper's.** Compare models within this app.
- **Token budget.** HELMET's shot counts were sized with the Llama-3.1 tokenizer. HELMET then cuts trailing context tokens to fit `length − 20 − 100`. Here prompts are counted with `o200k_base`, and whole trailing demos are dropped to fit `length − 2048` (the app's `RESERVE`, for the schema and Claude Code's overhead). At 16k and above every HELMET shot fits: 26k, 52k and 106k tokens at 32k, 64k and 128k. At 8k, 336 of 360 shots fit.
- **Test size.** 50 items instead of 500, to keep the quota and cost small. The intervals are about three times wider.
- **One of HELMET's five ICL datasets.** CLINC150, TREC coarse, TREC fine and NLU are left out, to keep the focus on banking, so this is not HELMET's ICL category average. The Kimi Linear paper doesn't say which HELMET-ICL subsets or decoding settings its 90.0 uses.
- **Decoding.** Temperature 0 through Instructor, where HELMET decodes greedily. Claude Code exposes no temperature setting.
- **Failed calls** (transport errors, or no valid JSON after retries) are shown, counted in `calls_failed` and left out of accuracy, as in the filing eval. A null or wrong label counts as wrong.

**Licences.** HELMET's code is MIT. BANKING77 is CC BY 4.0, per the [Hugging Face card](https://huggingface.co/datasets/PolyAI/banking77) and [PolyAI's repo](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE). It allows reuse with attribution: Casanueva et al. 2020. The data is cached, not committed. Run files keep each test item's text, which is attributed here.

## Cost to serve: memory per sequence

This is arithmetic from the published configs, not a measurement. It covers what an inference server holds **per sequence** for attention: the KV cache, which grows with length, plus the fixed state of the linear-attention layers. Weights are left out, since they are shared by every request.

Configs: [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) (revision `9c7f2fb`) and [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/blob/main/config.json) (revision `ac9c66c`). bf16 is 2 bytes. GB means 10^9 bytes. 32k, 128k and 262k mean 32,768, 131,072 and 262,144 tokens.

| | Qwen3-Next-80B-A3B | Qwen3-235B-A22B |
|---|---|---|
| layers | `num_hidden_layers` 48, `full_attention_interval` 4, so 12 full attention and 36 Gated DeltaNet | `num_hidden_layers` 94, all full attention |
| full-attention KV | `num_key_value_heads` 2, `head_dim` 256 | `num_key_value_heads` 4, `head_dim` 128 |
| linear-attention state | `linear_num_value_heads` 32 × `linear_key_head_dim` 128 × `linear_value_head_dim` 128, plus a conv state of (2 × `linear_num_key_heads` 16 × 128 + 32 × 128) = 8,192 channels × `linear_conv_kernel_dim` 4 | none |

The state shape `[num_v_heads, k_head_dim, v_head_dim]` and the conv width follow transformers' `modeling_qwen3_next.py`.

```
KV bytes per token = 2 (K and V) × full-attention layers × KV heads × head_dim × 2 bytes
  Qwen3-Next: 2 × 12 × 2 × 256 × 2 =  24,576 B  (24 KiB)
  Qwen3-235B: 2 × 94 × 4 × 128 × 2 = 192,512 B  (188 KiB), 7.8× more
Fixed state (Qwen3-Next) = 36 × (32 × 128 × 128 + 8,192 × 4) × 2 bytes = 40,108,032 B ≈ 0.040 GB
Per sequence = KV bytes per token × tokens + fixed state
```

| Tokens | Qwen3-Next KV | + fixed state | Qwen3-Next total | Qwen3-235B KV | Ratio |
|---|---|---|---|---|---|
| 32k | 0.81 GB | 0.04 GB | **0.85 GB** | **6.31 GB** | 7.5× |
| 128k | 3.22 GB | 0.04 GB | **3.26 GB** | **25.23 GB** | 7.7× |
| 262k | 6.44 GB | 0.04 GB | **6.48 GB** | **50.47 GB** | 7.8× |

If the recurrent state is kept in fp32, as some kernels do, it doubles to about 0.08 GB, which changes nothing above.

**What this shows.** At the same context length, one Qwen3-Next sequence needs about an eighth of the attention memory of one Qwen3-235B sequence. So a GPU with the same free memory fits about 7–8 times as many concurrent long sequences. Most of the saving comes from having 12 attention layers instead of 94. The fixed state equals the KV cache of about 1,600 tokens, so it is negligible at these lengths.

**What this doesn't show.** It's memory per sequence, not end-to-end serving cost or price. It leaves out the weights (80B against 235B total parameters, 3B against 22B active), prefill compute, kernel efficiency, batching and the scheduler, and what a provider charges: DeepInfra charges both models the same $0.09 per million input tokens. Qwen3-Next's remaining KV cache still grows linearly with length. Only the 36 linear layers are fixed. The two models also differ in size and training, so this is not a controlled architecture comparison.

**Measured speed, from the Kimi Linear paper** ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692), Figure 1(b) and Section 5.6). Kimi Linear's 3:1 KDA-to-MLA hybrid and a full-MLA baseline share the 48B layer and head layout. The hybrid reaches a **6.3× faster time per output token at 1M tokens (1.84 ms vs 11.48 ms)**, and 5.7× and 4.8× at shorter lengths on the same plot. The abstract says "up to 6 times decoding throughput for a 1M context" and a KV cache "reduced by up to 75%". The paper credits the TPOT gain partly to larger batches, which the smaller cache allows, so it is a throughput-under-memory-pressure result, not a single-request latency result. These are the paper's measurements on its own hardware, not ones made here.

## Files

- `app.py`: the web API and `main()`.
- `edgar.py`: EDGAR fetching and caching, HTML to text, passage finding, the exact and tolerant questions and ground truth.
- `scoring.py`: number and date parsing, exact match, outcomes (decoy included), stems, locating facts, Wilson intervals, the paired bootstrap and judge agreement.
- `context.py`: the planted design number, planted facts and their near and far decoys, truncation and token counting, and building the context sent.
- `llm.py`: `models.json`, the OpenRouter and Hugging Face price listings, JSON schema to pydantic, model calls (Instructor and Claude Code) with their latency, backoff on transient errors, extraction and the judge.
- `icl.py`: HELMET's ICL data (BANKING77 from Hugging Face), demo balancing, label remapping, test-item selection and prompt building.
- `runs.py`: presets, call modes, cost estimates, the run, retry and judge-check lifecycle, per-mode metrics, the paired headline, ICL scoring, and MLflow logging.
- `analyze.py`: summaries and figures from run files, offline: `uv run python analyze.py runs/<id>.json [...]` prints markdown tables (accuracy per mode and length, outcomes with near and far decoys, misses per filing, paired gaps; ICL accuracy and item agreement) and writes PNGs to `analysis/<run id>/`.
- `paths.py`: the shared paths.
- `index.html`: the single page.
- `models.json`: endpoints and models.
- `runs/`: one JSON file per run, committed because the post's figures are drawn from them.
- `mlflow.db`, `mlruns/`: the local MLflow store (git-ignored).
- `tests/`: normalisation, planting depth, near and far decoy placement, truncation, exact-match and outcome scoring (decoy and `decoy_hit` included), backoff on transport failures, retrying a run's failed calls, stems, passage finding and clipping, Wilson intervals, call modes in the plan and in scoring, the metrics (old runs included), the paired bootstrap, Hugging Face pricing and the judge-check agreement; for ICL, demo balancing, prompt budget, label remapping, test and demo separation, determinism, the plan's model lengths, the metrics and MLflow logging.
- `.cache/`: EDGAR responses and converted text, and the ICL parquet files under `.cache/icl/` (git-ignored).

Not in scope: auth, a database, deployment, PDF parsing, self-hosted models, and architecture speed or memory measurements, which an API can't show. The latency recorded per call includes the provider's queueing, and the memory figures above are arithmetic from configs.
