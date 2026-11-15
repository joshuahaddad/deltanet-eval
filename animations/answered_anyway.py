"""Real facts cut from the context (the right answer is "not in the text"): the full-attention Qwens answered half or
more of them anyway, the GDN hybrid 9%."""
from manim import *
import style

from figdata import absent, check

for cls in (Text, MathTex):
    cls.set_default(color=style.INK)

ROWS = [("Claude Sonnet 5.5", "Claude", style.MUTED), ("Qwen3-Next 80B-A3B", "GDN hybrid", style.ORANGE),
        ("Qwen3 30B-A3B", "30B", style.BLUE), ("Qwen3 235B-A22B", "235B", style.BLUE)]
said = {m: (len(a) - a.count("abstained"), len(a)) for m, a in absent.items()}
share = {m: check(said[m], "## 6.", m, 1) for m, _, _ in ROWS}
# the draft's claim 6: the GDN answered 9%, the 30B 47%, the 235B 65%
assert [f"{share[m]:.0%}" for m, _, _ in ROWS[1:]] == ["9%", "47%", "65%"]
gap = round((share["Qwen3 235B-A22B"] - share["Qwen3-Next 80B-A3B"]) * 100)
assert gap == 56

W = 8.4  # the width of 100%, in scene units
X0 = -W / 2  # bar start: the axis is centred, so the title and axis label sit over its middle
ROW_Y = [1.95, 0.6, -0.5, -1.6]
GDN_Y, BIG_Y = ROW_Y[1], ROW_Y[3]


class AnsweredAnyway(Scene):
    def construct(self):
        title = Text("What Percentage of 98 Queries Were Guessed Anyway?", font_size=style.BODY).set_y(3.4)
        grid = VGroup(*[VGroup(Line([X0 + a * W, 2.3, 0], [X0 + a * W, -2.75, 0], color=style.RULE, stroke_width=style.STRUCTURE),
                               Text(f"{a:.0%}", font_size=style.SMALL, color=style.MUTED).move_to([X0 + a * W, -3.05, 0]))
                        for a in (0, 0.5, 1)])
        x_lab = Text("questions answered instead of “not in the text”", font_size=style.SMALL, color=style.MUTED).set_y(-3.55)
        names, bars, labs = VGroup(), VGroup(), VGroup()
        for (m, name, c), y in zip(ROWS, ROW_Y):
            names.add(Text(name, font_size=style.SMALL, color=c).move_to([X0 - 0.3, y, 0], aligned_edge=RIGHT))
            bars.add(Rectangle(width=max(share[m] * W, 1e-3), height=0.6, fill_color=c, fill_opacity=style.BAR_OPACITY,
                               stroke_width=0).move_to([X0, y, 0], aligned_edge=LEFT))
            labs.add(Text(f"{share[m]:.0%}", font_size=style.SMALL, color=c).next_to(bars[-1], RIGHT, buff=0.2))

        # the gap: a brace over the GDN bar from its end to the 235B's value, a guide down to the 235B bar's end
        xg, xb = X0 + share["Qwen3-Next 80B-A3B"] * W, X0 + share["Qwen3 235B-A22B"] * W
        brace = BraceBetweenPoints([xg, GDN_Y + 0.35, 0], [xb, GDN_Y + 0.35, 0], UP, color=style.INK, buff=0.05)
        guide = DashedLine([xb, GDN_Y + 0.35, 0], [xb, BIG_Y + 0.3, 0], color=style.MUTED, stroke_width=style.STRUCTURE)
        gap_lab = Text(f"+{gap} pp", font_size=style.SMALL).next_to(brace, UP, buff=0.1)

        # poster: the question and the four models
        self.add(title, grid, x_lab, names)
        self.wait(1.5)
        self.play(*[GrowFromEdge(b, LEFT) for b in bars])
        self.play(FadeIn(labs))
        self.wait(1)
        self.play(Create(guide), GrowFromCenter(brace), FadeIn(gap_lab))
        self.wait(3)
