"""Team-strength model (Dixon-Coles / Maher Poisson regression).

Each international match contributes two observations:
    home goals ~ Poisson, attacker=home, defender=away, home=1
    away goals ~ Poisson, attacker=away, defender=home, home=0
Goals ~ C(attack) + C(defense) + home, fit by weighted Poisson GLM with
exponential time decay (recent matches weighted more). Fitting via statsmodels
gives coefficients WITH standard errors / p-values, which the variable-
selection step consumes directly.

A separate Dixon-Coles rho corrects the dependence in low scores (0-0,1-0,0-1,
1-1) for the joint scoreline distribution used in prediction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

WINDOW_YEARS = 15
MIN_MATCHES = 25          # teams below this are pooled into "Other"
DEFAULT_HALF_LIFE = 2.5   # years; time-decay so older form matters less


def _decay_weight(age_years: np.ndarray, half_life: float) -> np.ndarray:
    xi = np.log(2) / half_life
    return np.exp(-xi * age_years)


def build_long(matches: pd.DataFrame, ref_date: pd.Timestamp,
               half_life: float = DEFAULT_HALF_LIFE,
               team_whitelist: set | None = None) -> pd.DataFrame:
    """Long (two-row-per-match) frame for the Poisson GLM."""
    m = matches[matches["date"] <= ref_date].copy()
    m = m[m["date"] >= ref_date - pd.Timedelta(days=365.25 * WINDOW_YEARS)]
    age = (ref_date - m["date"]).dt.days / 365.25
    w = _decay_weight(age.to_numpy(), half_life)

    home = pd.DataFrame({
        "goals": m["home_score"].to_numpy(int),
        "attack": m["home_team"].to_numpy(),
        "defense": m["away_team"].to_numpy(),
        "home": np.where(m["neutral"].to_numpy(), 0, 1),
        "weight": w,
    })
    away = pd.DataFrame({
        "goals": m["away_score"].to_numpy(int),
        "attack": m["away_team"].to_numpy(),
        "defense": m["home_team"].to_numpy(),
        "home": 0,
        "weight": w,
    })
    long = pd.concat([home, away], ignore_index=True)

    if team_whitelist is None:
        counts = pd.concat([m["home_team"], m["away_team"]]).value_counts()
        team_whitelist = set(counts[counts >= MIN_MATCHES].index)
    long["attack"] = long["attack"].where(long["attack"].isin(team_whitelist), "Other")
    long["defense"] = long["defense"].where(long["defense"].isin(team_whitelist), "Other")
    return long, team_whitelist


class RatingModel:
    """Fitted strengths + a joint-scoreline predictor."""

    def __init__(self, glm_res, teams, rho, half_life):
        self.res = glm_res
        self.teams = set(teams) | {"Other"}
        teams = self.teams
        self.rho = rho
        self.half_life = half_life
        p = glm_res.params
        self.intercept = p.get("Intercept", 0.0)
        self.home_adv = p.get("home", 0.0)
        # attack/defense relative effects (reference team = 0)
        self.attack = {t: p.get(f"C(attack)[T.{t}]", 0.0) for t in teams}
        self.defense = {t: p.get(f"C(defense)[T.{t}]", 0.0) for t in teams}

    def _key(self, team: str) -> str:
        return team if team in self.teams else "Other"

    def expected_goals(self, team1: str, team2: str, neutral: bool = True):
        """(lambda1, lambda2): expected goals for team1 and team2."""
        t1, t2 = self._key(team1), self._key(team2)
        h = 0.0 if neutral else self.home_adv
        # defense coef is additive in the GLM (strong defenses are negative)
        log_l1 = self.intercept + self.attack[t1] + self.defense[t2] + h
        log_l2 = self.intercept + self.attack[t2] + self.defense[t1]
        return float(np.exp(log_l1)), float(np.exp(log_l2))

    def score_matrix(self, lam1: float, lam2: float, max_goals: int = 10):
        """Joint P(score1=i, score2=j) with Dixon-Coles low-score correction."""
        i = np.arange(max_goals + 1)
        p1 = poisson.pmf(i, lam1)
        p2 = poisson.pmf(i, lam2)
        M = np.outer(p1, p2)
        r = self.rho
        M[0, 0] *= 1 - lam1 * lam2 * r
        M[0, 1] *= 1 + lam1 * r
        M[1, 0] *= 1 + lam2 * r
        M[1, 1] *= 1 - r
        return M / M.sum()


def fit_dixon_coles_rho(matches: pd.DataFrame, ref_date: pd.Timestamp,
                        model: RatingModel, half_life: float) -> float:
    """MLE of rho on recent matches given fixed lambdas from the GLM."""
    m = matches[(matches["date"] <= ref_date) &
                (matches["date"] >= ref_date - pd.Timedelta(days=365.25 * 6))]
    lam1 = np.empty(len(m)); lam2 = np.empty(len(m)); x = m["home_score"].to_numpy(int); y = m["away_score"].to_numpy(int)
    for k, (_, row) in enumerate(m.iterrows()):
        lam1[k], lam2[k] = model.expected_goals(row["home_team"], row["away_team"], neutral=row["neutral"])
    low = (x <= 1) & (y <= 1)

    def neg_ll(rho):
        tau = np.ones(len(m))
        tau[low & (x == 0) & (y == 0)] = 1 - lam1[low & (x == 0) & (y == 0)] * lam2[low & (x == 0) & (y == 0)] * rho
        tau[low & (x == 0) & (y == 1)] = 1 + lam1[low & (x == 0) & (y == 1)] * rho
        tau[low & (x == 1) & (y == 0)] = 1 + lam2[low & (x == 1) & (y == 0)] * rho
        tau[low & (x == 1) & (y == 1)] = 1 - rho
        if np.any(tau <= 0):
            return 1e9
        return -np.sum(np.log(tau))

    out = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
    return float(out.x)


def fit(matches: pd.DataFrame, ref_date: pd.Timestamp,
        half_life: float = DEFAULT_HALF_LIFE) -> RatingModel:
    long, teams = build_long(matches, ref_date, half_life)
    glm = smf.glm("goals ~ C(attack) + C(defense) + home", data=long,
                  family=sm.families.Poisson(), freq_weights=long["weight"]).fit()
    model = RatingModel(glm, teams, rho=0.0, half_life=half_life)
    model.rho = fit_dixon_coles_rho(matches, ref_date, model, half_life)
    return model


if __name__ == "__main__":
    from .data import load_internationals
    intl = load_internationals()
    ref = pd.Timestamp("2026-06-11")
    model = fit(intl, ref)
    print(f"teams in model: {len(model.teams)}  home_adv(log)={model.home_adv:.3f}  rho={model.rho:.3f}")
    # strongest attacks
    top = sorted(model.attack.items(), key=lambda kv: -kv[1])[:10]
    print("top attack ratings:", [(t, round(v, 2)) for t, v in top])
    l1, l2 = model.expected_goals("Argentina", "Mexico", neutral=True)
    print(f"Argentina vs Mexico expected goals: {l1:.2f} - {l2:.2f}")
