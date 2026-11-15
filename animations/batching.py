"""128k and 192k, one planted code per call vs all 15 in one call: batching lifts full attention by 24 to 51 points,
while the GDN hybrid gains little, and dips at 128k (on only 5 batched calls)."""
from manim import *
import style

from figdata import acc, by, cell, check, length_label, planted

for cls in (Text, MathTex):
    cls.set_default(color=style.INK)

GDN, BIG, SMALL_FA = "Qwen3-Next 80B-A3B", "Qwen3 235B-A22B", "Qwen3 30B-A3B"
ROWS = [(GDN, "GDN hybrid", style.ORANGE), (BIG, "235B", style.BLUE), (SMALL_FA, "30B", style.BLUE)]
ALL = sorted({r["length"] for r in planted(by[GDN], "single")})
LONG = [L for L in ALL if L >= 131072]
rows = lambda m, mode, L: [r for r in planted(by[m], mode) if r["length"] == L]
v = {(m, mode, L): check(acc(rows(m, mode, L)), f"**{mode}**", m, 1 + ALL.index(L))
     for m, _, _ in ROWS for mode in ("single", "batched") for L in LONG}
gain = {(m, L): v[m, "batched", L] - v[m, "single", L] for m, _, _ in ROWS for L in LONG}
for (m, L), g in gain.items():
    assert f"{g:+.0%}" == cell("Batching effect", m, 1 + ALL.index(L)), (m, L, g)
calls = len({r["ticker"] for r in rows(GDN, "batched", LONG[0])})  # a batched call is one filing at one length
assert [length_label(L) for L in LONG] == ["128k", "192k"] and calls == 5
assert f"{v[GDN, 'batched', LONG[0]]:.0%}" == "68%"  # the draft's claim 2

LO, HI, W = 0.4, 1.0, 4.2  # x range of each panel, in accuracy and in scene units
PANEL_X = [-3.4, 2.0]      # left edge of each panel
ROW_Y = [1.0, -0.3, -1.6]


class Batching(Scene):
    def construct(self):
        title = Text("Batched questions rescue full attention, not the hybrid", font_size=style.BODY).set_y(3.4)
        key = VGroup(Circle(0.09, color=style.INK, stroke_width=3), Text("one code per call", font_size=style.SMALL),
                     Arrow(ORIGIN, 0.8 * RIGHT, buff=0, color=style.INK, stroke_width=3, max_tip_length_to_length_ratio=0.3),
                     Dot(radius=0.1, color=style.INK), Text("all 15 codes in one call", font_size=style.SMALL))
        key.arrange(RIGHT, buff=0.2).set_y(2.6)
        x = lambda p, a: PANEL_X[p] + (a - LO) / (HI - LO) * W

        frame = VGroup()
        for p, L in enumerate(LONG):
            frame.add(Text(length_label(L), font_size=style.BODY).move_to([x(p, 0.7), 1.85, 0]))
            for a in (0.5, 0.75, 1.0):
                frame.add(Line([x(p, a), 1.5, 0], [x(p, a), -2.1, 0], color=style.RULE, stroke_width=style.STRUCTURE),
                          Text(f"{a:.0%}", font_size=style.SMALL, color=style.MUTED).move_to([x(p, a), -2.45, 0]))
        names = VGroup(*[Text(name, font_size=style.SMALL, color=c).move_to([PANEL_X[0] - 0.35, y, 0], aligned_edge=RIGHT)
                         for (_, name, c), y in zip(ROWS, ROW_Y)])

        single, move = VGroup(), {}
        for (m, _, c), y in zip(ROWS, ROW_Y):
            for p, L in enumerate(LONG):
                a, b = [x(p, v[m, mode, L]) for mode in ("single", "batched")]
                single.add(Circle(0.11, color=c, stroke_width=3).move_to([a, y, 0]))
                g = round(100 * gain[m, L])
                lab = Text(f"{g:+d}".replace("-", "−"), font_size=style.SMALL, color=c)
                lab.next_to([b, y, 0], RIGHT if b > a else LEFT, buff=0.2)
                move[m, L] = VGroup(Line([a, y, 0], [b, y, 0], buff=0.11, color=c, stroke_width=4), Dot([b, y, 0], radius=0.12, color=c), lab)

        # poster: everything at one code per call
        self.add(title, key, frame, names, single)
        self.wait(1.5)
        self.play(*[FadeIn(move[m, L]) for m in (BIG, SMALL_FA) for L in LONG])
        self.wait(1.5)
        self.play(*[FadeIn(move[GDN, L]) for L in LONG])
        self.wait(3)
