"""<Topic> in a few lines of NumPy: the reference for the post's claims.

One example, no batching: the point is the idea, not speed. Run this file to check every claim the post makes.
Scenes import from here (the repo root is on PYTHONPATH in the figure-kit image).
"""
import numpy as np


def core(x):
    """The method the post explains, written as plainly as the maths."""
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4)
    # One assert per claim in the post, with the post's own numbers.
    assert np.allclose(core(x), x), "claim: <what the post says>"
    print("all claims check out")
