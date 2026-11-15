"""One planted code per call: every model fades with length, full attention to under half by 192k, the GDN hybrid to 80%."""
from manim import *
import style

from figdata import acc, by, check, length_label, planted

for cls in (Text, MathTex):
    cls.set_default(color=style.INK)

GDN, BIG, SMALL_FA = "Qwen3-Next 80B-A3B", "Qwen3 235B-A22B", "Qwen3 30B-A3B"
LENGTHS = sorted({r["length"] for r in planted(by[GDN], "single")})
COLS = {L: 1 + i for i, L in enumerate(LENGTHS)}  # summary.md column of each length
y = {m: [check(acc([r for r in planted(by[m], "single") if r["length"] == L]), "**single**", m, COLS[L]) for L in LENGTHS]
     for m in (GDN, BIG, SMALL_FA)}
n = [len([r for r in planted(by[GDN], "single") if r["length"] == L]) for L in LENGTHS]
# the draft's claim 4: 99% -> 80% for the GDN, 93% -> 47% for 30B, 100% -> 49% for 235B
assert [f"{y[m][0]:.0%}->{y[m][-1]:.0%}" for m in (GDN, SMALL_FA, BIG)] == ["99%->80%", "93%->47%", "100%->49%"]
assert [length_label(L) for L in LENGTHS] == ["8k", "32k", "128k", "192k"] and n == [150, 135, 75, 45]


class SingleByLength(Scene):
    def construct(self):
        title = Text("One code per call: full attention falls below half by 192k", font_size=style.BODY).set_y(3.4)
        ax = Axes(x_range=[0, 3, 1], y_range=[0, 1, 0.25], x_length=7.8, y_length=4.6, tips=False,
                  axis_config={"color": style.MUTED, "stroke_width": style.STRUCTURE})
        ax.shift([-5.0, -2.6, 0] - ax.c2p(0, 0))
        grid = VGroup(*[Line(ax.c2p(0, v), ax.c2p(3, v), color=style.RULE, stroke_width=style.STRUCTURE) for v in (0.25, 0.5, 0.75, 1)])
        yt = VGroup(*[Text(f"{v:.0%}", font_size=style.SMALL).next_to(ax.c2p(0, v), LEFT, buff=0.2) for v in (0, 0.5, 1)])
        xt = VGroup(*[Text(length_label(L), font_size=style.SMALL).next_to(ax.c2p(i, 0), DOWN, buff=0.2) for i, L in enumerate(LENGTHS)])
        x_lab = Text("context length (tokens)", font_size=style.SMALL, color=style.MUTED).next_to(xt, DOWN, buff=0.15)
        y_lab = Text("codes recalled", font_size=style.SMALL, color=style.MUTED).rotate(PI / 2).next_to(yt, LEFT, buff=0.2)

        def line(m, color):  # [dot at 8k, (segment, dot) per later length]
            pts = [ax.c2p(i, v) for i, v in enumerate(y[m])]
            dot = lambda p: Dot(p, radius=0.08, color=color)
            return [dot(pts[0])] + [VGroup(Line(a, b, color=color, stroke_width=style.DATA), dot(b)) for a, b in zip(pts, pts[1:])]
        lines = {GDN: line(GDN, style.ORANGE), BIG: line(BIG, style.BLUE), SMALL_FA: line(SMALL_FA, style.BLUE)}

        end = lambda m: ax.c2p(3, y[m][-1])
        lab_gdn = Text(f"GDN hybrid  {y[GDN][-1]:.0%}", font_size=style.SMALL, color=style.ORANGE).next_to(end(GDN), RIGHT, buff=0.25)
        # 235B and 30B end 2 points apart: stack their labels and point each at its line
        lab_big = Text(f"235B  {y[BIG][-1]:.0%}", font_size=style.SMALL, color=style.BLUE).move_to(end(BIG) + [0.25, 0.35, 0], aligned_edge=LEFT)
        lab_small = Text(f"30B  {y[SMALL_FA][-1]:.0%}", font_size=style.SMALL, color=style.BLUE).move_to(end(SMALL_FA) + [0.25, -0.4, 0], aligned_edge=LEFT)
        fa = Text("full attention", font_size=style.SMALL, color=style.BLUE).next_to(VGroup(lab_big, lab_small), DOWN, buff=0.2, aligned_edge=LEFT)

        # poster: title, axes and the 8k start points (all near 100%)
        self.add(title, grid, ax, yt, xt, x_lab, y_lab, *[l[0] for l in lines.values()])
        self.wait(1)
        for i in range(1, len(LENGTHS)):  # one length at a time, all three models together
            self.play(*[Create(l[i]) for l in lines.values()])
        self.play(FadeIn(lab_gdn), FadeIn(lab_big), FadeIn(lab_small), FadeIn(fa))
        self.wait(3)
