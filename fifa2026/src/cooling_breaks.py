"""Cooling-break spike analysis.

Question: after a cooling break, does anything measurable change -- do goals or
momentum spike? We only have open goal-minute data (possession/xG are not
reachable under GitHub-only egress), so we test what the data supports:

  1. Scoring-rate discontinuity: goals/min in a window AFTER the break vs an
     equal window BEFORE it. Conditional-Poisson (binomial) exact test: given
     the total goals in pre+post, the post count is Binomial(N, 0.5) under the
     null of no rate change. Effect = post/pre rate ratio.
  2. Momentum / "who scores next": after a break, is the next goal scored by
     the team that was trailing (a comeback spike) more than chance?

Break schedule:
  * 2026  -> mandatory ~22' of each half  => break minutes 22 and 67
  * 2014-2022 -> heat-triggered ~30'/75'  => break minutes 30 and 75
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import binomtest

BREAKS = {2014: (30, 75), 2018: (30, 75), 2022: (30, 75), 2026: (22, 67)}
WINDOW = 8  # minutes on each side of a break


def goal_table(wc: pd.DataFrame) -> pd.DataFrame:
    """Long per-goal table: year, match_id, minute, side (1/2)."""
    rows = []
    for mid, (_, m) in enumerate(wc.iterrows()):
        for side, col in ((1, "goals1_min"), (2, "goals2_min")):
            for mins in m[col]:
                if not np.isnan(mins):
                    rows.append({"year": m["year"], "match_id": mid,
                                 "minute": float(mins), "side": side})
    return pd.DataFrame(rows)


def _count(goals: pd.DataFrame, years, lo, hi) -> int:
    """Goals in (lo, hi] minutes across the given years."""
    g = goals[goals["year"].isin(years)]
    return int(((g["minute"] > lo) & (g["minute"] <= hi)).sum())


def rate_discontinuity(goals: pd.DataFrame, years, break_min: int, w: int = WINDOW):
    """Compare equal pre/post windows around a break minute."""
    pre = _count(goals, years, break_min - w, break_min)
    post = _count(goals, years, break_min, break_min + w)
    n = pre + post
    p = binomtest(post, n, 0.5).pvalue if n else 1.0
    rr = (post / pre) if pre else float("inf")
    return {"break": break_min, "pre": pre, "post": post, "rate_ratio": rr, "p": p}


def momentum_next_goal(wc: pd.DataFrame, break_min: int, side_select):
    """After `break_min`, which team scores next: leading, level or trailing?"""
    lead = level = trail = 0
    for _, m in wc.iterrows():
        if m["year"] not in side_select:
            continue
        # cumulative score at the break minute
        s1 = sum(1 for x in m["goals1_min"] if not np.isnan(x) and x <= break_min)
        s2 = sum(1 for x in m["goals2_min"] if not np.isnan(x) and x <= break_min)
        # next goal strictly after the break
        after = [(x, 1) for x in m["goals1_min"] if not np.isnan(x) and x > break_min]
        after += [(x, 2) for x in m["goals2_min"] if not np.isnan(x) and x > break_min]
        if not after:
            continue
        after.sort()
        scorer = after[0][1]
        scorer_lead = (s1 - s2) if scorer == 1 else (s2 - s1)
        if scorer_lead > 0:
            lead += 1
        elif scorer_lead == 0:
            level += 1
        else:
            trail += 1
    return lead, level, trail


def analyze(wc: pd.DataFrame) -> str:
    goals = goal_table(wc)
    out = []
    out.append("# Cooling-break spike analysis\n")
    out.append(f"Window = +/-{WINDOW} min around each break. "
               "Test: post-break goal count ~ Binomial(pre+post, 0.5) under no-change null.\n")

    groups = {
        "2026 (mandatory breaks ~22'/67', n=%d matches)" % (wc.year == 2026).sum(): ([2026], BREAKS[2026]),
        "2014-2022 (heat breaks ~30'/75', n=%d matches)" % (wc.year != 2026).sum(): ([2014, 2018, 2022], BREAKS[2014]),
    }
    for label, (years, brks) in groups.items():
        out.append(f"\n## {label}")
        pooled_pre = pooled_post = 0
        for bm in brks:
            r = rate_discontinuity(goals, years, bm)
            pooled_pre += r["pre"]; pooled_post += r["post"]
            rr = r["rate_ratio"]
            out.append(f"- break {bm}': pre={r['pre']} post={r['post']} "
                       f"rate_ratio={rr:.2f} p={r['p']:.3f}")
        n = pooled_pre + pooled_post
        pooled_p = binomtest(pooled_post, n, 0.5).pvalue if n else 1.0
        rr = pooled_post / pooled_pre if pooled_pre else float("inf")
        verdict = "SIGNIFICANT" if pooled_p < 0.05 else "not significant"
        out.append(f"- **pooled both breaks**: pre={pooled_pre} post={pooled_post} "
                   f"rate_ratio={rr:.2f} p={pooled_p:.3f} -> {verdict}")

    out.append("\n## Momentum: who scores next after a break")
    for label, (years, brks) in {"2026": ([2026], BREAKS[2026]),
                                  "2014-2022": ([2014, 2018, 2022], BREAKS[2014])}.items():
        L = V = T = 0
        for bm in brks:
            l, v, t = momentum_next_goal(wc, bm, set(years))
            L += l; V += v; T += t
        decisive = L + T
        p = binomtest(T, decisive, 0.5).pvalue if decisive else 1.0
        out.append(f"- {label}: next scorer was leading={L}, level={V}, trailing={T} "
                   f"(trailing vs leading p={p:.3f})")

    out.append("\n## Interpretation")
    out.append("- A rate_ratio > 1 with p < 0.05 means scoring genuinely spikes after the break;")
    out.append("  < 1 means a lull (rhythm disruption). Absent significance, the viewer-felt")
    out.append("  rhythm change does NOT translate into a measurable goal-rate effect, so per")
    out.append("  the 'significant variables only' rule it is excluded from the predictor.")
    out.append("- Possession/xG are unavailable under current egress; this tests goal events only.")
    return "\n".join(out)


if __name__ == "__main__":
    import os
    from .data import load_worldcup
    wc = load_worldcup()
    report = analyze(wc)
    print(report)
    path = os.path.join(os.path.dirname(__file__), "..", "reports", "cooling_break_analysis.md")
    with open(path, "w") as f:
        f.write(report + "\n")
