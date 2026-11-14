# TITLE, explained

The companion repo for the post [*POST TITLE*](POST_URL).

- `reference.py`: the idea in a few lines of NumPy, with checks for the post's claims.
- `animations/`: the post's figures, one [Manim](https://www.manim.community/) scene per file. The shared look comes from [figure-kit](https://github.com/joshuahaddad/figure-kit).
- `figures/`: the rendered GIFs.
- `eval/`: filing-recall, a small webapp that measures exact versus tolerant recall on long SEC filings, comparing a Gated DeltaNet hybrid (Qwen3-Next) with full-attention models. It's self-contained (its own `pyproject.toml`), so it can move to its own repo later.
- `interactive/index.html`: an interactive version you can play with in the browser ([live page](https://joshuahaddad.github.io/deltanet-eval/interactive/)).

## Run

Check the math (needs [uv](https://docs.astral.sh/uv/)):

```bash
uv run reference.py
```

Render the figures (needs Docker). The first run builds the pinned figure-kit image.

```bash
./make_animations.sh          # all of them
./make_animations.sh <name>   # just one
```

## The core idea

```python
# the two or three lines the post is about
```

## Papers

- <Title>. [arXiv:NNNN.NNNNN](https://arxiv.org/abs/NNNN.NNNNN)
