"""1st-half / 2nd-half scoring split.

The international model yields full-time expected goals (lambda_ft). To produce
1H / 2H / FT outputs we split lambda_ft into the two halves using the empirical
share of goals scored in each half, estimated from World Cup HT/FT scores
(authoritative half counts, no stoppage-time ambiguity).

Well-known regularity: more goals fall in the 2nd half (fatigue, substitutions,
chasing the game). We measure it rather than assume it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def half_goal_counts(wc: pd.DataFrame) -> pd.DataFrame:
    """Per match: goals in each half from HT/FT (both teams combined and split)."""
    d = wc.dropna(subset=["ht1", "ht2"]).copy()
    d["g1_1h"] = d["ht1"]; d["g2_1h"] = d["ht2"]
    d["g1_2h"] = d["ft1"] - d["ht1"]; d["g2_2h"] = d["ft2"] - d["ht2"]
    d["tot_1h"] = d["g1_1h"] + d["g2_1h"]
    d["tot_2h"] = d["g1_2h"] + d["g2_2h"]
    return d


def estimate_half_shares(wc: pd.DataFrame):
    """(share_1h, share_2h) of total goals, with a per-year breakdown."""
    d = half_goal_counts(wc)
    g1 = d["tot_1h"].sum(); g2 = d["tot_2h"].sum(); tot = g1 + g2
    overall = (g1 / tot, g2 / tot)
    by_year = {}
    for yr, grp in d.groupby("year"):
        a, b = grp["tot_1h"].sum(), grp["tot_2h"].sum()
        by_year[int(yr)] = (a / (a + b), b / (a + b))
    return overall, by_year


class HalfModel:
    """Splits full-time lambdas into per-half lambdas."""

    def __init__(self, share_1h: float):
        self.s1 = share_1h
        self.s2 = 1 - share_1h

    def split(self, lam1_ft: float, lam2_ft: float):
        """Return dict of per-half expected goals for both teams."""
        return {
            "1H": (lam1_ft * self.s1, lam2_ft * self.s1),
            "2H": (lam1_ft * self.s2, lam2_ft * self.s2),
            "FT": (lam1_ft, lam2_ft),
        }


def fit_half_model(wc: pd.DataFrame) -> HalfModel:
    (s1, _), _ = estimate_half_shares(wc)
    return HalfModel(s1)


if __name__ == "__main__":
    from .data import load_worldcup
    wc = load_worldcup()
    overall, by_year = estimate_half_shares(wc)
    print(f"overall 1H/2H goal share: {overall[0]:.3f} / {overall[1]:.3f}")
    print("by year:")
    for yr, (a, b) in by_year.items():
        print(f"  {yr}: 1H={a:.3f}  2H={b:.3f}")
