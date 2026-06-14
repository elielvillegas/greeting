"""Variable selection by statistical significance.

The requirement: include ONLY variables with a mathematically demonstrated,
statistically significant effect. We test each candidate against the data we can
actually reach (GitHub-only egress) and keep / drop accordingly. Honest outcome:
team strength, home advantage and the half effect are demonstrable; altitude,
heat and cooling breaks are NOT demonstrable from available data, so they are
excluded rather than asserted.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .data import load_internationals, load_worldcup
from .ratings import build_long
from .halves import half_goal_counts


def test_team_strength(intl, ref):
    """LR test: do team attack/defense effects matter beyond intercept+home?"""
    long, _ = build_long(intl, ref)
    full = smf.glm("goals ~ C(attack) + C(defense) + home", data=long,
                   family=sm.families.Poisson(), freq_weights=long["weight"]).fit()
    reduced = smf.glm("goals ~ home", data=long,
                      family=sm.families.Poisson(), freq_weights=long["weight"]).fit()
    lr = reduced.deviance - full.deviance
    df = int(reduced.df_resid - full.df_resid)
    p = stats.chi2.sf(lr, df)
    return {"name": "Team attack/defense strength", "stat": f"LR chi2={lr:.0f}, df={df}",
            "p": p, "keep": p < 0.05}, full


def test_home(full):
    """Wald test on the home-advantage coefficient."""
    beta = full.params["home"]; p = full.pvalues["home"]
    return {"name": "Home / host advantage", "stat": f"beta={beta:.3f} (x{np.exp(beta):.2f} goals)",
            "p": p, "keep": p < 0.05}


def test_half(wc):
    """Poisson GLM: do 2nd-half goal counts differ from 1st-half?"""
    d = half_goal_counts(wc)
    stacked = pd.DataFrame({
        "goals": np.r_[d["tot_1h"].to_numpy(), d["tot_2h"].to_numpy()],
        "half": ["1H"] * len(d) + ["2H"] * len(d),
    })
    res = smf.glm("goals ~ C(half)", data=stacked, family=sm.families.Poisson()).fit()
    beta = res.params["C(half)[T.2H]"]; p = res.pvalues["C(half)[T.2H]"]
    return {"name": "Half (2H vs 1H scoring)", "stat": f"beta={beta:.3f} (x{np.exp(beta):.2f})",
            "p": p, "keep": p < 0.05}


def test_stage(wc):
    """Poisson GLM: do knockout matches score differently from group games?"""
    g = wc.copy()
    g["goals"] = g["ft1"] + g["ft2"]
    res = smf.glm("goals ~ C(stage)", data=g, family=sm.families.Poisson()).fit()
    key = [k for k in res.params.index if k.startswith("C(stage)")][0]
    beta = res.params[key]; p = res.pvalues[key]
    return {"name": "Stage (knockout vs group)", "stat": f"beta={beta:.3f} (x{np.exp(beta):.2f})",
            "p": p, "keep": p < 0.05}


def report():
    intl = load_internationals()
    wc = load_worldcup()
    ref = pd.Timestamp("2026-06-11")

    rows = []
    team_row, full = test_team_strength(intl, ref)
    rows.append(team_row)
    rows.append(test_home(full))
    rows.append(test_half(wc))
    rows.append(test_stage(wc))
    # Not testable from available data -> cannot demonstrate -> excluded
    rows.append({"name": "Venue altitude", "stat": "no altitude variation in 2014-2026 WC venues",
                 "p": None, "keep": False})
    rows.append({"name": "Venue heat index", "stat": "per-match weather not reachable (egress)",
                 "p": None, "keep": False})
    rows.append({"name": "Cooling-break post-break effect", "stat": "see cooling_break_analysis.md (pooled p=1.00)",
                 "p": None, "keep": False})

    lines = ["# Variable significance selection\n",
             "Keep iff a statistically significant effect is demonstrated (p<0.05).\n",
             "| Variable | Evidence | p-value | Decision |",
             "|---|---|---|---|"]
    for r in rows:
        p = "n/a" if r["p"] is None else (f"{r['p']:.2e}" if r["p"] < 1e-3 else f"{r['p']:.3f}")
        lines.append(f"| {r['name']} | {r['stat']} | {p} | {'KEEP' if r['keep'] else 'DROP'} |")
    lines.append("\n**Kept:** " + ", ".join(r["name"] for r in rows if r["keep"]))
    lines.append("\n**Dropped (not demonstrably significant from available data):** "
                 + ", ".join(r["name"] for r in rows if not r["keep"]))
    lines.append("\nNote: time-decay weighting (recent form) is a fitting choice tuned by "
                 "out-of-sample likelihood in backtest.py, not a hypothesis test.")
    text = "\n".join(lines)
    path = os.path.join(os.path.dirname(__file__), "..", "reports", "variable_significance.md")
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


if __name__ == "__main__":
    print(report())
