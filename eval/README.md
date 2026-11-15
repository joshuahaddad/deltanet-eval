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
export <provider keys>=...        # the Qwen models use HF_TOKEN via Hugging Face, (Claude runs through your local `claude` login)
export EDGAR_CONTACT=you@example.com # SEC requires a contact in the User-Agent
uv run app.py                        # http://127.0.0.1:8780 (set PORT to change)
uv run pytest
```

| Env var | Used for |
|---|---|
| `HF_TOKEN` | Required if `models.json` has huggingface models. A fine-grained token with "Make calls to Inference Providers". |
| `OPENROUTER_API_KEY` | Required if `models.json` has an OpenRouter model  |
| `LITELLM_API_KEY` | Required if `models.json` has a LiteLLM gateway entry |
| `ANTHROPIC_API_KEY` | optional: when set, `claude -p` bills it instead of your Claude subscription |
| `EDGAR_CONTACT` | Your email, the `User-Agent: filing-recall <contact>` header that EDGAR requires |
| `PORT` | optional override for the app port, default 8780 |
| `MLFLOW_TRACKING_URI` | optional: your MLflow server, e.g. `http://host:5000`. Unset logs to the local `eval/mlflow.db`. |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | optional: for a server with auth turned on (MLflow reads them itself) |
| `MLFLOW_EXPERIMENT_NAME` | optional, default `filing-recall` |

Keys are read from the environment only and are never written to runs or logs.

On the page:

**Run smoke** will run a very small test before attemtping to run the full grid. A few calls to the models to test they work.

**Run full grid…** will run the full test on ten FY2024 filings (AAPL, MSFT, GE, KO, F, PFE, BAC, JPM, C and WFC) at various context lengths and modes.

  | Length | Filings | Count |
  |---|---|---|
  | 8k | all ten | 10 |
  | 32k | all but WFC | 9 |
  | 128k | F, PFE, BAC, JPM, C (KO, at 126k tokens, is just short) | 5 |
  | 192k | BAC, JPM, C | 3 |

## Models

`models.json` lists the models under test, the judge and, under `judge_check`, an optional second judge (see [Judge check](#judge-check)). Every model listed runs in every run. Each entry is one of two kinds:

- **OpenAI-compatible endpoint:** `base_url`, `api_key_env` and `model`. Structured output goes through [Instructor](https://python.useinstructor.com/) in JSON mode, which handles parsing and retries. OpenRouter's catalogue, or the Hugging Face router's listing for the provider after the `:` suffix, supplies context length and price. The Hugging Face router reports tokens but no cost, so the cost is computed from that listed price. For any other endpoint, add `"context_length"`. When the router doesn't list the model's provider, or lists it with no price, add `"context_length"` and `"pricing": {"input": ..., "output": ...}` in dollars per million tokens, from the provider's own page.
- **`"provider": "claude-code"`:** `model` and `context_length`. The call runs the local `claude -p` with no tools, no settings, no MCP servers and `--json-schema`, so it uses your Claude Code login (your subscription). If `ANTHROPIC_API_KEY` is set, Claude Code bills that key instead. Claude Code enforces the schema itself, and pydantic validates the result. Under a subscription, the run records `"billing": "subscription"`, leaves `cost` empty and puts the list-price figure in `notional_cost`. The full grid then uses up subscription quota, not money.

A LiteLLM gateway, for local models, is an OpenAI-compatible entry. The `model` is the gateway's model name, and `context_length` is what the gateway reports as `max_input_tokens`:

```json
{"name": "Local coder", "base_url": "http://localhost:4000/v1", "api_key_env": "LITELLM_API_KEY",
 "model": "local-coder", "context_length": 131072}
```

## MLflow

Each finished run is logged to MLflow. By default it goes to a local store, `eval/mlflow.db` with artifacts in `eval/mlruns/`, both git-ignored. Browse it with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To use your server instead, set `MLFLOW_TRACKING_URI`, plus the username and password if it uses auth.

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


**Planted facts.** Near 10, 30, 50, 70 and 90% of the context that is sent, the app inserts three paragraphs, 2% apart, each with a made-up reference code (exact, for example `QX-48213-7`) and a short made-up statement (tolerant). That gives 15 exact and 15 tolerant planted questions per filing and length. Values are seeded per filing, so every model, length and mode sees the same ones.

**Decoys.** Each planted code has two near-duplicates that are never asked about: the same kind of paragraph for a lookalike subject, with a different code and a filler sentence.

- **Near:** 0.5% after its target, in the same depth bin ("the Brookfield archive survey" against the asked "Brookfield archive audit"). It sits halfway between the target and the next 2%-spaced plant or far decoy, so it never shares their line.
- **Far:** 40 points deeper, wrapped into the 10–90% bins (10 to 50, 30 to 70, 50 to 90, 70 to 10, 90 to 30), plus 1% ("the Brookfield archive review"), so it falls between the 2%-spaced plants.

**Planted tolerant answers are scored by stems, not by the judge.** Each planted statement has one or two stems, and a correct answer must contain every one as a lowercase substring. `a|b` accepts either: for "drivers reported fatigue", the stems are `driver` and `fatigu|tired|exhaust`. This is cheap, deterministic and needs no judge, but it can miss a correct paraphrase (see [Design choices and limits](#design-choices-and-limits)).

**Call modes.** Each filing and length is asked in more than one way, all with the same context:

- `batched`: every question in one call, as a practitioner would extract a filing. Real and planted questions are asked together.
- `single`: one call per planted code, 15 per filing and length, so no question can interfere with another. Planted notes and real questions are only batched. Notes were at the ceiling in single mode in the [first run](#first-run) (96–100%), so single calls for them spent money without telling the models apart. Real questions would multiply the cost.
- `closed_book`: the real questions with an empty document, once per filing, to measure what the model knows from memory. Its instruction differs from the open-book one: it says the filing is not shown, asks for an answer from memory, and the request names the company and fiscal year ("The filing is AAPL's 10-K for fiscal year 2024"). The open-book instruction says to answer from the document only, which would force null. Closed-book tolerant answers are recorded but not judged.

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

The judge (Claude Haiku through Claude Code) is an LLM, so its verdicts need checking. **Check judge** on a finished run re-grades a fixed sample of up to 60 real tolerant answers, seeded by the run ID, with the `judge_check` model in `models.json`. As mentioned in the previous section, the judge only scores non-stem graded gist answers.

## Design choices and limits

- **The bootstrap resamples items, not filings.** Questions from one filing are correlated, so the paired intervals are narrower than they should be. With 10 filings, a bootstrap over filings would be too coarse to be useful. It is worth switching once there are a few dozen filings.
- **10 filings, fewer at length.** Every per-filing effect (layout, length, how the company writes) rests on ten documents at 8k, five at 128k and three at 192k. Three of the five 128k filings, and all three at 192k, are banks, so the long end leans on one kind of filing.
- **Retries change the timing, not the context.** A retried call sends the same context, rebuilt from the code, but at a later time and maybe to a less busy provider.
- **Stems can miss paraphrases.** A correct planted note answer that uses none of a stem's words ("the trucks' crews were worn out") counts as wrong. Stems trade that for a deterministic score with no judge in the loop.
- **Single mode asks codes only.** So there is no single-mode tolerant panel, and batching cost is measured on codes only.
- **Latency / Cost**. Since the models were used through API by different providers, we cannot compare latency or cost as a reflection of the architecture, it is recorded as a diagnostic.

## HELMET ICL mode

**What it measures.** [HELMET](https://github.com/princeton-nlp/HELMET) (Yen et al., ICLR 2025, [arXiv:2410.02694](https://arxiv.org/abs/2410.02694)) has a many-shot in-context learning category. The prompt holds hundreds to thousands of labelled examples (`text`, then `label: N`), followed by one unlabelled text, and the model must give its label. Labels are shown as **random integers**, not intent names, so the model can't lean on what "card_arrival" means. It has to learn the mapping from the examples in the context. This mode runs HELMET's banking dataset, [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) (77 banking intents), to keep the focus on the financial domain.

**Why it's the gist case.** No single example holds the answer, and no exact string has to be copied out. The model needs a summary of what each label's examples have in common, built up over the whole context. A fixed-size recurrent state can hold that kind of summary, so the post's claim predicts that a linear-attention hybrid keeps up with full attention here, while it falls behind on the exact panels above. The Kimi Linear paper ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692), Table 5, models trained on 1.4T tokens, at 128k context) reports HELMET-ICL at 90.0 for Kimi Linear, 88.0 for full-attention MLA and 85.5 for the Gated DeltaNet hybrid (GDN-H). show the hybrid is good at exact recall, and a gap between this mode and the planted exact codes is what the post's claim predicts.

**Presets.**

| Preset | Datasets | Lengths | Test items | Models | Calls |
|---|---|---|---|---|---|
| `icl-smoke` | BANKING77 | 32k | 5 | every model | 15 |
| `icl` | BANKING77 | 32k, 64k, 128k | 50 | every model; Claude at 32k only | 350 |

One call is one test item at one length. A preset's `model_lengths` limits a model, by its `name` or its `provider`, to some lengths. `icl` sets `{"claude-code": [32768]}`, because Claude runs on subscription quota, and so does the filing preset `full`.
