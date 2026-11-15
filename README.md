# Teach linear attention to forget, and word order comes free

The companion repo for [*Teach linear attention to forget, and word order comes free*](POST_URL), part 2 of the DeltaNet series ([part 1](https://joshuahaddad.substack.com/p/drop-softmax-and-attention-becomes-an-rnn)).

The post adds forget gates to the delta rule: Gated DeltaNet's per-head decay, Kimi Delta Attention's per-channel decay, and Gated DeltaNet-2's separate erase and write gates. It shows why that recurrence also encodes word order, so Kimi Linear can drop RoPE. It then tests a Gated DeltaNet hybrid (Qwen3-Next 80B-A3B) against full-attention Qwen3 30B-A3B and 235B-A22B on 10-K filings.

Headline results, from a small run:

- Asked one planted code per call at 192k tokens, the hybrid got 80% and the full-attention models 47–49%.
- When the needed fact had been cut from the context, the hybrid answered anyway 9% of the time and the full-attention models 47–65%.
- With every question in one call, full attention caught up and mostly pulled ahead.

## The core idea

```python
S = alpha * S                      # forget: Gated DeltaNet uses one scalar per head, KDA one rate per key channel (alpha[:, None])
S += beta * np.outer(k, v - k @ S) # delta rule: read what k holds, nudge it toward v
o = q @ S
```

## What's here

- `reference.py`: memory per sequence for the three Qwen models, with checks for every number in the post's KV cache table.
- `eval/`: filing-recall, the benchmark webapp (10-K planted codes with decoys, gist recall, hallucination, closed-book, HELMET ICL BANKING77). It is self-contained, with its own `pyproject.toml` and [README](eval/README.md).
- `animations/`: the post's figures, one [Manim](https://www.manim.community/) scene per file. The shared look comes from [figure-kit](https://github.com/joshuahaddad/figure-kit).
- `figures/`: the rendered GIFs and PNGs.

## Run

These commands need [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv run reference.py           # check the memory table
./make_animations.sh          # render all figures (first run builds the figure-kit image)
./make_animations.sh <name>   # render one
```

To run the benchmark, follow [eval/README.md](eval/README.md).

## Papers

- Gated Delta Networks: Improving Mamba2 with Delta Rule. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Kimi Linear: An Expressive, Efficient Attention Architecture. [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
- Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention. [arXiv:2605.22791](https://arxiv.org/abs/2605.22791)
- HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly. [arXiv:2410.02694](https://arxiv.org/abs/2410.02694)
