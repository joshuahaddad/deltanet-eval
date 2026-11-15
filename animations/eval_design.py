"""The eval on one card: each 10-K cut to four context lengths with codes planted at the same five depths, and one
question whose arrows land on the planted code (right) or one of its two look-alike decoys (wrong)."""
from manim import *
import style

from figdata import by, length_label
from context import DEPTHS, planted_facts
from runs import PRESETS

for cls in (Text, MathTex):
    cls.set_default(color=style.INK)

LENGTHS = PRESETS["full"]["lengths"]
assert DEPTHS == (0.1, 0.3, 0.5, 0.7, 0.9)
assert [length_label(L) for L in LENGTHS] == ["8k", "32k", "128k", "192k"]
row = next(r for r in by["Qwen3-Next 80B-A3B"] if r["field"] == "planted_code_50a")  # the codes the model saw
plant = next(p for p in planted_facts("any") if p["field"] == "planted_code_50a")  # depths and subjects
subject = plant["question"].removeprefix("What is the internal reference code for ").removesuffix("?")
near_subject = plant["decoy_paragraphs"]["near"].split(": ")[0].removeprefix("Internal reference code for ")
far_subject = plant["decoy_paragraphs"]["far"].split(": ")[0].removeprefix("Internal reference code for ")
assert row["truth"] and set(row["decoys"]) == {"near", "far"} and subject in row["question"]
pos = {"planted": plant["target"], **plant["decoy_targets"]}
assert pos == {"planted": 0.48, "near": 0.485, "far": 0.89}

XL, W, H = -5.4, 11.9, 0.3  # bars: left edge, width of the longest cut, height
ROWS = [2.2, 1.65, 1.1, 0.55]  # one row per length, shortest on top


class EvalDesign(Scene):
    def construct(self):
        title = Text("10-K Recall Test: Planted Code Retrieval With Decoys", font_size=style.BODY).set_y(3.4)
        bars, longest = VGroup(), None
        for L, y in zip(LENGTHS, ROWS):
            w = W * L / LENGTHS[-1]
            bar = Rectangle(width=w, height=H, fill_color=style.SURFACE, fill_opacity=1, color=style.MUTED,
                            stroke_width=style.STRUCTURE).move_to([XL, y, 0], aligned_edge=LEFT)
            ticks = VGroup(*[Line([XL + d * w, y - H / 2, 0], [XL + d * w, y + H / 2, 0], color=style.TARGET,
                                  stroke_width=3) for d in DEPTHS])
            lab = Text(length_label(L), font_size=style.SMALL, color=style.MUTED).next_to(bar, LEFT, buff=0.2)
            lab.align_to([XL - 0.2, 0, 0], RIGHT)
            bars.add(VGroup(bar, ticks, lab))
            longest = (bar, w, y)
        bar, w, y = longest
        depth = VGroup(*[Text(f"{d:.0%}", font_size=style.SMALL, color=style.TARGET).move_to([XL + d * w, y - 0.45, 0])
                         for d in DEPTHS])
        legend = Text("context lengths, to scale · 15 codes at the same depths in each", font_size=style.SMALL,
                      color=style.MUTED).move_to([0, 2.8, 0])
        # the example's three positions on the longest cut: the planted code and both decoys
        mark = lambda d, c: Line([XL + d * w, y - H / 2 - 0.08, 0], [XL + d * w, y + H / 2 + 0.08, 0], color=c,
                                 stroke_width=style.DATA)
        marks = VGroup(mark(pos["near"], style.ERROR), mark(pos["far"], style.ERROR), mark(pos["planted"], style.TARGET))

        q = Text(f"Q: What is the internal reference code for {subject}?", font_size=style.SMALL).set_y(-0.75)

        def card(sub, code, note, c, cx):
            body = RoundedRectangle(width=4.3, height=2.0, corner_radius=0.12, fill_color=style.SURFACE, fill_opacity=1,
                                    color=c, stroke_width=style.STRUCTURE).move_to([cx, -2.85, 0])
            ws = sub.split()
            half = len(ws) // 2
            words = VGroup(Paragraph(" ".join(ws[:half]), " ".join(ws[half:]), font_size=style.SMALL, color=style.MUTED,
                                     alignment="center"), Text(code, font_size=style.SMALL, color=c),
                           Text(note, font_size=style.SMALL, color=style.MUTED)).arrange(DOWN, buff=0.12).move_to(body)
            return VGroup(body, words)
        cards = VGroup(card(subject, f"✓ {row['truth']}", f"planted at {pos['planted']:.0%}", style.TARGET, -4.55),
                       card(near_subject, f"✗ {row['decoys']['near']}", f"decoy at {pos['near']:.1%}", style.ERROR, 0),
                       card(far_subject, f"✗ {row['decoys']['far']}", f"decoy at {pos['far']:.0%}", style.ERROR, 4.55))
        for c in cards:
            assert c.width <= 4.35, f"card text overflows: {c.width:.2f} {[w.width for w in c[1]]}"
        start = q.get_bottom() + 0.1 * DOWN
        arrows = VGroup(Arrow(start, cards[0].get_top(), buff=0.08, color=style.TARGET, stroke_width=style.DATA,
                              tip_length=0.22, max_tip_length_to_length_ratio=0.5),
                        *[DashedVMobject(Arrow(start, c.get_top(), buff=0.08, color=style.ERROR, stroke_width=style.STRUCTURE + 1,
                                               tip_length=0.2, max_tip_length_to_length_ratio=0.5), num_dashes=max(6, round(np.linalg.norm(c.get_top() - start) * 5)))
                          for c in cards[1:]])

        self.add(title, legend, bars, depth, marks, q, arrows, cards)
