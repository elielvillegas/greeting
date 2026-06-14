"""Per-match predictor: scoring, result and winner for 1H / 2H / FT.

Uses only the variables demonstrated significant in selection.py: team
attack/defense strength, home advantage, and the half split. Cooling-break /
heat / altitude were tested and dropped, so they do not enter here.

  python -m src.predict --team1 Argentina --team2 Mexico
  python -m src.predict --team1 Mexico --team2 USA --host Mexico   # host gets home edge
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from .data import load_internationals, load_worldcup
from .ratings import fit
from .halves import fit_half_model


def outcome(matrix: np.ndarray):
    """(p_team1_win, p_draw, p_team2_win) from a joint scoreline matrix."""
    p1 = np.tril(matrix, -1).sum()   # score1 > score2
    pd_ = np.trace(matrix)
    p2 = np.triu(matrix, 1).sum()
    return float(p1), float(pd_), float(p2)


def most_likely(matrix: np.ndarray):
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(i), int(j), float(matrix[i, j])


def summarize(model, lam1, lam2, label):
    M = model.score_matrix(lam1, lam2)
    p1, pdraw, p2 = outcome(M)
    i, j, pml = most_likely(M)
    over25 = sum(M[a, b] for a in range(M.shape[0]) for b in range(M.shape[1]) if a + b >= 3)
    btts = M[1:, 1:].sum()
    return {
        "label": label, "xg1": lam1, "xg2": lam2,
        "p1": p1, "pdraw": pdraw, "p2": p2,
        "ml": (i, j, pml), "over25": float(over25), "btts": float(btts),
    }


def predict(team1, team2, host=None, max_goals=10, continuity=None):
    intl = load_internationals()
    wc = load_worldcup()
    ref = intl["date"].max()
    model = fit(intl, ref, continuity=continuity)
    half = fit_half_model(wc)

    neutral = host not in (team1, team2)
    # expected_goals applies home advantage to team1 when not neutral, so order by host
    if host == team2:
        l2, l1 = model.expected_goals(team2, team1, neutral=False)
    else:
        l1, l2 = model.expected_goals(team1, team2, neutral=neutral)

    splits = half.split(l1, l2)
    res = {ph: summarize(model, splits[ph][0], splits[ph][1], ph) for ph in ("1H", "2H", "FT")}
    return team1, team2, host, neutral, res


def render(team1, team2, host, neutral, res):
    site = "neutral venue" if neutral else f"host advantage: {host}"
    lines = [f"\n{team1}  vs  {team2}    ({site})",
             "=" * 62,
             f"{'Phase':6} {'xG':>11}  {'1-win':>6} {'draw':>6} {'2-win':>6}  {'Likely':>7} {'O2.5':>6} {'BTTS':>6}",
             "-" * 62]
    for ph in ("1H", "2H", "FT"):
        r = res[ph]
        xg = f"{r['xg1']:.2f}-{r['xg2']:.2f}"
        ml = f"{r['ml'][0]}-{r['ml'][1]}"
        lines.append(f"{ph:6} {xg:>11}  {r['p1']*100:5.1f}% {r['pdraw']*100:5.1f}% "
                     f"{r['p2']*100:5.1f}%  {ml:>7} {r['over25']*100:5.1f}% {r['btts']*100:5.1f}%")
    ft = res["FT"]
    winner = team1 if ft["p1"] > ft["p2"] else team2
    conf = max(ft["p1"], ft["p2"]) * 100
    lines.append("-" * 62)
    lines.append(f"Full-time pick: {winner}  (win prob {conf:.1f}%, draw {ft['pdraw']*100:.1f}%)")
    lines.append(f"Most likely FT score: {team1} {ft['ml'][0]}-{ft['ml'][1]} {team2} "
                 f"(p={ft['ml'][2]*100:.1f}%)")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team1", required=True)
    ap.add_argument("--team2", required=True)
    ap.add_argument("--host", default=None, help="team playing at home (gets host advantage)")
    ap.add_argument("--rosters", default=None,
                    help="path to filled 2026 rosters JSON; discounts changed-squad history")
    args = ap.parse_args()
    cont = None
    if args.rosters:
        from .rosters import continuity_scores, load_current_rosters
        cont = continuity_scores(load_current_rosters(args.rosters))
        print(f"[roster continuity applied for {len(cont)} teams]")
    out = predict(args.team1, args.team2, host=args.host, continuity=cont)
    print(render(*out))
