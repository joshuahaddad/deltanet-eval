"""A stored value's strength over 300 tokens: GDN decays a whole head at one rate (α = 0.99); KDA gives each channel its
own, so a slow channel (α = 0.999) keeps a cover-page fact while a fast one (α = 0.97) drops the boilerplate."""
import numpy as np
from manim import *
import style

for cls in (Text, MathTex, Tex):
    cls.set_default(color=style.INK)
MATH = 34  # Tex words draw larger than Text; 34 matches style.SMALL by eye

T = 300
GDN, SLOW, FAST = 0.99, 0.999, 0.97
half = lambda a: np.log(0.5) / np.log(a)  # tokens until a value written once is down to half strength
assert [round(half(a)) for a in (GDN, SLOW, FAST)] == [69, 693, 23]
assert np.isclose(SLOW ** T, 0.741, atol=5e-4) and FAST ** T < 1e-3 and np.isclose(GDN ** T, 0.049, atol=5e-4)


class ForgetRates(Scene):
    def construct(self):
        title = Text("GDN: one forget rate per head. KDA: one per channel", font_size=style.BODY).set_y(3.4)

        def panel(x0, head):
            ax = Axes(x_range=[0, T, 100], y_range=[0, 1, 0.5], x_length=5.2, y_length=3.8, tips=False,
                      axis_config={"color": style.MUTED, "stroke_width": style.STRUCTURE})
            ax.shift([x0, -2.4, 0] - ax.c2p(0, 0))
            mid = DashedLine(ax.c2p(0, 0.5), ax.c2p(T, 0.5), color=style.RULE, stroke_width=style.STRUCTURE)
            ticks = VGroup(*[Text(str(t), font_size=style.SMALL, color=style.MUTED).next_to(ax.c2p(t, 0), DOWN, buff=0.15)
                             for t in (0, 100, 200, 300)],
                           *[Text(s, font_size=style.SMALL, color=style.MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.15)
                             for s, v in (("half", 0.5), ("full", 1))])
            lab = Text(head, font_size=style.BODY).next_to(ax.c2p(T / 2, 1), UP, buff=0.35)
            return ax, VGroup(ax, mid, ticks, lab)

        axl, left = panel(-5.4, "Gated DeltaNet")
        axr, right = panel(1.4, "Kimi Delta Attention")
        x_lab = Text("tokens since the value was written", font_size=style.SMALL, color=style.MUTED).set_y(-3.55)
        curve = lambda ax, a, c: ax.plot(lambda t: a ** t, x_range=[0, T, 1], color=c, stroke_width=style.DATA)
        g, s, f = curve(axl, GDN, style.ORANGE), curve(axr, SLOW, style.ORANGE), curve(axr, FAST, style.MUTED)

        def mark(ax, a, c):  # dot where the curve crosses half strength, and its half-life
            return Dot(ax.c2p(half(a), 0.5), radius=0.08, color=c)
        g_lab = Tex(rf"whole head, $\alpha = {GDN}$\\half-life {half(GDN):.0f} tokens", font_size=MATH, color=style.ORANGE)
        g_lab.next_to(axl.c2p(half(GDN), 0.5), UR, buff=0.2)
        s_lab = Tex(rf"cover page (slow), $\alpha = {SLOW}$\\half-life {half(SLOW):.0f} tokens", font_size=MATH,
                    color=style.ORANGE).move_to(axr.c2p(T, 0.66), aligned_edge=UR)
        f_lab = Tex(rf"boilerplate (fast), $\alpha = {FAST}$\\half-life {half(FAST):.0f} tokens", font_size=MATH,
                    color=style.MUTED).move_to(axr.c2p(T, 0.08), aligned_edge=DR)

        # poster: both empty panels and the title
        self.add(title, left, right, x_lab)
        self.wait(1.5)
        self.play(Create(g), Create(s), Create(f), run_time=4, rate_func=linear)
        self.play(FadeIn(mark(axl, GDN, style.ORANGE)), FadeIn(g_lab))
        self.wait(1)
        self.play(FadeIn(s_lab))
        self.play(FadeIn(mark(axr, FAST, style.MUTED)), FadeIn(f_lab))
        self.wait(3)
