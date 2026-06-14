"""Out-of-sample validation.

Train on internationals strictly before a tournament, predict its matches, and
score the W/D/L probabilities with proper scoring rules:
  * log-loss            (lower better)
  * multiclass Brier    (lower better)
  * RPS (ranked prob.)  (lower better; the standard for ordered football W/D/L)
Compared against a uniform 1/3-1/3-1/3 baseline. Also reports top-pick accuracy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .data import load_internationals, load_worldcup
from .ratings import fit
from .predict import outcome

HOSTS = {2014: {"Brazil"}, 2018: {"Russia"}, 2022: {"Qatar"},
         2026: {"United States", "Mexico", "Canada"}}


def result_vec(s1, s2):
    """One-hot [team1 win, draw, team2 win]."""
    if s1 > s2: return np.array([1, 0, 0])
    if s1 == s2: return np.array([0, 1, 0])
    return np.array([0, 0, 1])


def rps(p, y):
    """Ranked probability score for 3 ordered categories."""
    cp, cy = np.cumsum(p), np.cumsum(y)
    return np.sum((cp[:-1] - cy[:-1]) ** 2) / (len(p) - 1)


def backtest_year(year: int):
    intl = load_internationals()
    wc = load_worldcup()
    games = wc[wc["year"] == year].copy()
    start = games["date"].min()
    train = intl[intl["date"] < start]
    model = fit(train, start - pd.Timedelta(days=1))

    P, Y = [], []
    for _, m in games.iterrows():
        host = HOSTS.get(year, set())
        if m["team1"] in host:
            l1, l2 = model.expected_goals(m["team1"], m["team2"], neutral=False)
        elif m["team2"] in host:
            l2, l1 = model.expected_goals(m["team2"], m["team1"], neutral=False)
        else:
            l1, l2 = model.expected_goals(m["team1"], m["team2"], neutral=True)
        M = model.score_matrix(l1, l2)
        P.append(np.array(outcome(M)))
        Y.append(result_vec(m["ft1"], m["ft2"]))
    P, Y = np.array(P), np.array(Y)
    eps = 1e-12
    logloss = -np.mean(np.log((P * Y).sum(1) + eps))
    brier = np.mean(((P - Y) ** 2).sum(1))
    rps_m = np.mean([rps(P[i], Y[i]) for i in range(len(P))])
    acc = np.mean(P.argmax(1) == Y.argmax(1))

    base = np.full(3, 1 / 3)
    bl_logloss = -np.mean(np.log((base * Y).sum(1) + eps))
    bl_brier = np.mean(((base - Y) ** 2).sum(1))
    bl_rps = np.mean([rps(base, Y[i]) for i in range(len(Y))])
    return {"year": year, "n": len(P), "logloss": logloss, "brier": brier,
            "rps": rps_m, "acc": acc, "bl_logloss": bl_logloss,
            "bl_brier": bl_brier, "bl_rps": bl_rps}


def main():
    print(f"{'Tourney':8} {'n':>4} | {'logloss':>8} {'brier':>7} {'rps':>7} {'acc':>6} "
          f"| {'base_LL':>8} {'base_brier':>10} {'base_rps':>8}")
    print("-" * 84)
    for yr in (2018, 2022, 2026):
        try:
            r = backtest_year(yr)
        except Exception as e:
            print(f"{yr}: skipped ({e})"); continue
        print(f"{yr:<8} {r['n']:>4} | {r['logloss']:8.3f} {r['brier']:7.3f} {r['rps']:7.3f} "
              f"{r['acc']*100:5.1f}% | {r['bl_logloss']:8.3f} {r['bl_brier']:10.3f} {r['bl_rps']:8.3f}")
    print("\nModel beats the uniform baseline when logloss/brier/rps are LOWER than base_*.")


if __name__ == "__main__":
    main()
