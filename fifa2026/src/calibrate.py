"""Probability calibration (temperature scaling).

The raw model is overconfident on World Cups (favorites lose more often than
their probabilities imply -- the 2022 upsets). Temperature scaling softens the
distribution: q_k proportional to p_k**(1/T), with T>1 flattening toward the
prior. T is chosen to minimise out-of-sample log-loss over past World Cups, then
applied at prediction time. Saved to cache/calibration.json.
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.optimize import minimize_scalar

import pandas as pd
from .data import load_internationals
from .ratings import fit
from .predict import outcome
from .backtest import result_vec

CAL_PATH = os.path.join(os.path.dirname(__file__), "..", "cache", "calibration.json")

# Major final tournaments give a representative, upset-inclusive calibration set
# (broader than World Cups alone, so no single tournament dominates T).
MAJORS = ("FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations")


def _temper(P, T):
    Q = np.power(P, 1.0 / T)
    return Q / Q.sum(axis=1, keepdims=True)


def collect(since="2012-01-01"):
    """Walk-forward over major final tournaments: train strictly before each one,
    predict its matches out-of-sample, pool (P, Y)."""
    intl = load_internationals()
    majors = intl[intl["tournament"].isin(MAJORS) & (intl["date"] >= since)].copy()
    majors["edition"] = majors["tournament"] + " " + majors["date"].dt.year.astype(str)
    Ps, Ys = [], []
    for _, games in majors.groupby("edition"):
        start = games["date"].min()
        if len(games) < 8:
            continue
        model = fit(intl[intl["date"] < start], start - pd.Timedelta(days=1), with_rho=False)
        for _, m in games.iterrows():
            # hosts of finals tournaments vary; treat as neutral for calibration
            l1, l2 = model.expected_goals(m["home_team"], m["away_team"], neutral=True)
            Ps.append(np.array(outcome(model.score_matrix(l1, l2))))
            Ys.append(result_vec(m["home_score"], m["away_score"]))
    return np.array(Ps), np.array(Ys)


def fit_temperature(save=True):
    P, Y = collect()
    eps = 1e-12

    def logloss(T):
        Q = _temper(P, T)
        return -np.mean(np.log((Q * Y).sum(1) + eps))

    raw = logloss(1.0)
    out = minimize_scalar(logloss, bounds=(0.5, 4.0), method="bounded")
    T = float(out.x)
    result = {"T": T, "logloss_raw": float(raw), "logloss_cal": float(logloss(T))}
    if save:
        with open(CAL_PATH, "w") as f:
            json.dump(result, f, indent=2)
    return result


def load_temperature(default=1.0):
    if os.path.exists(CAL_PATH):
        with open(CAL_PATH) as f:
            return json.load(f).get("T", default)
    return default


def apply_temperature(p, T=None):
    """Temper a single probability vector (list/array)."""
    if T is None:
        T = load_temperature()
    q = np.power(np.asarray(p, float), 1.0 / T)
    return q / q.sum()


if __name__ == "__main__":
    r = fit_temperature()
    print(f"Fitted temperature T = {r['T']:.3f}")
    print(f"log-loss  raw={r['logloss_raw']:.3f}  ->  calibrated={r['logloss_cal']:.3f}  "
          f"({100*(r['logloss_raw']-r['logloss_cal'])/r['logloss_raw']:.1f}% better)")
