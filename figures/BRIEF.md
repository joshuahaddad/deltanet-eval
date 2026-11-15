# Shared brief: figures for DeltaNet, part 2 (forget gates, and a practical benchmark)

Fill this in once per article and save it as `figures/BRIEF.md`. Every designer and coder agent gets it.

**The post:** part 2 of the DeltaNet series: forget gates (Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2), positional encoding moving into memory, and a small long-context benchmark of Qwen3-Next (a Gated DeltaNet hybrid) against full-attention Qwen3 models and Claude. The text is at `Professional/Writing/Drafts/09 Draft.md` in the vault.

**The reader:** a software or ML engineer who knows the original transformer in outline and has read part 1 (RNN form, delta rule, crosstalk, fixed-size state vs KV cache). They read on a phone or laptop, in a white or dark column about 680 px wide. Many only glance at the figures, so each GIF has to make one point that lands without reading the paragraph.

**The author's brand:** how theory connects to engineering. Plain and concrete, no hype.

**Medium:** an animated GIF made with Manim CE v0.19.0 through figure-kit. `./make_animations.sh <name>` turns `animations/<name>.py` into `figures/<name>.gif` (or `figures/<name>.png` when the scene has no animation). One Scene class per file. `-qm` (720p) at 15 fps.

**Style:** follow figure-kit's `STYLE.md` (dark plate, ink + plum, CMU Serif + Euler). `import style` and use its constants; never `WHITE`, never opacity to dim.
- Colour meaning for this post, carried over from part 1 (orange = the delta rule's fixed-size state `S`, sky = the plain sum / the growing KV cache):
  - `PRIMARY` (orange) = the Gated DeltaNet hybrid, Qwen3-Next 80B-A3B (fixed-size state).
  - `SECONDARY` (sky) = full attention, Qwen3 235B-A22B and Qwen3 30B-A3B (KV cache). Both share the hue; tell them apart with direct labels, never a legend.
  - `MUTED` = Claude Sonnet 5.5, a reference run only at 32k, and context.
  - `TARGET` (mint) = correct / the answer; `ERROR` (coral, with a ✗) = wrong or a guess.
- Progressive highlighting: one piece at a time (`style.box`, `style.dim`, Transform).
- First frame not blank (Substack and email show it as the poster). End on a 2 to 3 s held still.
- 8 to 20 s, under 1 MB if possible, hard cap 5 MB.
- Labels at `style.SMALL` or larger. Title = the takeaway. Direct labels at line ends, no legends.
- Notation as in the article and part 1: `S`, `k`, `v`, `q`, `α` (forget gate), `β` (write strength). Model names as the draft writes them: "GDN hybrid", "full attention", "30B", "235B".
- Numbers: only those in the article, from `eval/runs/*.json` through `eval/analyze.py` via `figdata.py` (the repo root is on `PYTHONPATH`). Each scene computes its numbers and asserts them against `summary.md` with `figdata.check`.

**Simplicity:** each scene is one short, readable file with no shared framework beyond `style` and `figdata`.
