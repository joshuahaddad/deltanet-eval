"""15 codes planted in a 192k filing: one question's code with a decoy just after it and one far after, then a late code whose far decoy comes first."""
from math import isclose

from manim import *
import style

from figdata import by, length_label
from context import planted_facts
from runs import PRESETS

for cls in (Text, MathTex):
    cls.set_default(color=style.INK)

LENGTHS = PRESETS["full"]["lengths"]
assert length_label(LENGTHS[-1]) == "192k"
facts = {p["field"]: p for p in planted_facts("any")}
codes = [p["target"] for p in facts.values() if p["field"].startswith("planted_code")]  # all 15, 2 points apart
assert len(codes) == 15


def case(field):
    """The codes the model saw (row), the depths (plant), and the subjects of the code and its decoys."""
    row = next(r for r in by["Qwen3-Next 80B-A3B"] if r["field"] == field)
    plant = facts[field]
    subj = plant["question"].removeprefix("What is the internal reference code for ").removesuffix("?")
    sub = {k: p.split(": ")[0].removeprefix("Internal reference code for ") for k, p in plant["decoy_paragraphs"].items()}
    assert row["truth"] and set(row["decoys"]) == {"near", "far"} and subj in row["question"]
    assert subj.split()[:-1] == sub["near"].split()[:-1] == sub["far"].split()[:-1]  # only the last word differs
    assert plant["target"] in codes
    return row, {"planted": plant["target"], **plant["decoy_targets"]}, subj, sub


row, pos, subject, sub = case("planted_code_50a")  # case 1: both decoys after the code
assert pos == {"planted": 0.48, "near": 0.485, "far": 0.89}
row2, pos2, subject2, sub2 = case("planted_code_90b")  # case 2: the far decoy wraps round to before the code
assert (pos2["planted"], pos2["far"]) == (0.90, 0.31)

FADED_T = interpolate_color(ManimColor(style.TARGET), ManimColor(style.BG), 0.45)  # other questions' answers
XL, W, H, Y = -5.4, 11.9, 0.4, 1.0  # the bar: left edge, width, height, centre
BOT, TOP, ARROW = Y - H / 2, Y + H / 2, 1.75  # the arrow runs above the bar
PY, PH = -0.3, 0.5  # zoom panel centre and height
x = lambda d: XL + W * d  # a depth on the bar
zx = lambda d: (d - 0.485) * 300  # a depth in the zoom panel: x -3..3 covers 47.5..49.5%
assert isclose(x(.475), 0.25, abs_tol=0.005) and isclose(x(.495), 0.49, abs_tol=0.005)
assert isclose(zx(.48), -1.5) and isclose(zx(.485), 0, abs_tol=1e-9)
MARK, FAINT = 3, 2  # every green/red line, every faded tick
vline = lambda x0, y0, y1, c, w=MARK: Line([x0, y0, 0], [x0, y1, 0], color=c, stroke_width=w)
arrow = lambda a, b: Arrow([x(a), ARROW, 0], [x(b), ARROW, 0], buff=0, color=style.ERROR, stroke_width=MARK,
                           tip_length=0.2, max_tip_length_to_length_ratio=0.5)


def question(subj):
    q = Text(f"Q: What is the internal reference code for {subj}?", font_size=style.SMALL,
             t2c={subj: style.TARGET}).set_y(2.85)
    return q, VGroup(*[c for c in q if c.get_color() == ManimColor(style.TARGET)])


def stack(lines, edge, ax, top, slot):
    """Label lines (text, colour[, t2c]) at a 0.42 pitch, aligned on one edge."""
    g = VGroup(*[Text(t, font_size=style.SMALL, color=c, t2c=m[0] if m else None).set_y(top - 0.15 - 0.42 * i)
                 for i, (t, c, *m) in enumerate(lines)])
    for t in g:
        t.align_to([ax, 0, 0], edge)
    assert g.width <= slot, f"{lines[0][0]!r} stack {g.width:.2f} > slot {slot:.2f}"
    return g


def labels(head, code, subj, c):  # the subject in INK, only its differing last word coloured
    return [(head, c), (code, c), (subj, style.INK, {subj.split()[-1]: c})]


class EvalDesign(Scene):
    def construct(self):
        title = Text("Exact Recall Test With Decoys", font_size=style.BODY).set_y(3.45)
        q, phrase = question(subject)
        q.set_color(style.INK)
        bar = Rectangle(width=W, height=H, fill_color=style.SURFACE, fill_opacity=1, color=style.MUTED,
                        stroke_width=style.STRUCTURE).move_to([XL, Y, 0], aligned_edge=LEFT)
        size = Text(length_label(LENGTHS[-1]), font_size=style.SMALL, color=style.MUTED).set_y(Y).align_to([-5.6, 0, 0], RIGHT)
        ticks = {d: vline(x(d), BOT, TOP, FADED_T, FAINT) for d in codes}
        edge = lambda box: box.copy().set_fill(opacity=0).set_z_index(2)  # box outlines drawn over the lines that meet them
        self.add(title, q, bar, edge(bar), size)
        self.wait(0.8)

        # B1: the 15 planted codes; their label sits above the bar's left end, clear of both arrows
        note = Text("15 planted codes", font_size=style.SMALL, color=style.MUTED).move_to([0, ARROW, 0]).align_to([XL, 0, 0], LEFT)
        assert note.get_right()[0] < x(pos2["far"]) - 0.3 and note.get_top()[1] < q.get_bottom()[1] - 0.2
        self.play(LaggedStart(*[Create(t) for t in ticks.values()], lag_ratio=0.1), run_time=1.0, rate_func=style.SNAPPY)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(0.4)

        # B2: the question picks one
        case1 = VGroup(vline(x(.48), TOP, ARROW, style.TARGET))  # case 1's annotations, cleared in B6
        self.play(phrase.animate.set_color(style.TARGET), ticks[.48].animate.set_color(style.TARGET).set_stroke(width=MARK), Create(case1[0]), run_time=0.8)
        self.wait(0.8)

        # B3: zoom on 47.5-49.5%, the answer
        window = Rectangle(width=0.24, height=H, color=style.INK, stroke_width=style.STRUCTURE).move_to([0.37, Y, 0]).set_z_index(3)
        panel = Rectangle(width=6, height=PH, fill_color=style.SURFACE, fill_opacity=1, color=style.MUTED,
                          stroke_width=style.STRUCTURE).move_to([0, PY, 0])
        ptop, pbot = PY + PH / 2, PY - PH / 2
        links = VGroup(*[Line([wx, BOT - 0.06, 0], [px, ptop, 0], color=style.MUTED, stroke_width=style.STRUCTURE)
                         for wx, px in ((0.25, -3), (0.49, 3))])
        zoom = VGroup(window, links, panel, edge(panel), vline(zx(.48), ptop, pbot, style.TARGET))
        self.play(Create(zoom), run_time=1)
        answer = VGroup(vline(zx(.48), pbot, -1.25, style.TARGET),
                        stack(labels("the answer", f"✓ {row['truth']}", subject, style.TARGET), RIGHT, -1.65, -1.1, -1.65 + 6.9))
        self.play(FadeIn(answer), run_time=0.4)
        self.wait(1)

        # B4: the nearby decoy, half a point later
        zn = vline(zx(.485), ptop, pbot, style.ERROR)
        self.play(Create(zn), run_time=0.5)
        near = VGroup(vline(zx(.485), pbot, -1.25, style.ERROR),
                      stack(labels("nearby decoy", f"✗ {row['decoys']['near']}", sub["near"], style.ERROR),
                            LEFT, 0.15, -1.1, x(.89) - 0.15 - 0.15))  # stays clear of the far leader
        self.play(FadeIn(near), run_time=0.4)
        self.wait(1.1)

        # B5: the distant decoy, 41 points later; one ERROR line from the stub top down to its label
        fmark, fwd = vline(x(.89), ARROW, BOT, style.ERROR), arrow(.48, .89)
        self.play(Create(fmark), run_time=0.5)
        self.play(GrowArrow(fwd), run_time=0.8)
        far = VGroup(vline(x(.89), BOT, -2.45, style.ERROR),
                     stack(labels("distant decoy", f"✗ {row['decoys']['far']}", sub["far"], style.ERROR), RIGHT, 6.6, -2.6, 6.6 + 6.9))
        assert far.get_bottom()[1] > -3.85, far.get_bottom()
        self.play(FadeIn(far), run_time=0.4)
        self.wait(1.5)

        # B6: case 2, a code at 90% whose distant decoy wraps round to 31%, before it
        a, d = pos2["planted"], pos2["far"]
        q2, phrase2 = question(subject2)
        case1.add(zoom, answer, zn, near, fmark, fwd, far)
        self.play(FadeOut(case1), ticks[.48].animate.set_color(FADED_T).set_stroke(width=FAINT), Transform(q, q2), run_time=1)
        self.play(ticks[a].animate.set_color(style.TARGET).set_stroke(width=MARK), Create(vline(x(a), TOP, ARROW, style.TARGET)),
                  Create(vline(x(d), ARROW, BOT, style.ERROR)), run_time=0.6)
        self.play(GrowArrow(arrow(a, d)), run_time=0.8)
        # both stacks hang below the bar, each leader ending just above its stack; the decoy's sits left of the answer's
        ans2 = VGroup(vline(x(a), BOT, -0.35, style.TARGET),
                      stack(labels("the answer", f"✓ {row2['truth']}", subject2, style.TARGET), RIGHT, 6.6, -0.5, 6.6 + 6.9))
        dec2 = VGroup(vline(x(d), BOT, -0.35, style.ERROR),
                      stack(labels("distant decoy", f"✗ {row2['decoys']['far']}", sub2["far"], style.ERROR), RIGHT, x(d) + 1.0, -0.5, x(d) + 1.0 + 6.9))
        assert dec2.get_right()[0] < ans2[1].get_left()[0] - 0.3 and dec2.get_left()[0] > -6.9, (dec2.get_right(), ans2[1].get_left())
        self.play(FadeIn(ans2), FadeIn(dec2), run_time=0.4)

        self.wait(2.8)  # the still
