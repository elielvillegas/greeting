"""Full 2026 schedule access + batch prediction for a given date.

load_worldcup() keeps only *played* matches; the raw schedule also holds future
fixtures, but with placeholder slots (group codes '1A'/'3A/B/...', knockout
'W73'/'L101'). We read the full schedule, keep real country matchups, normalise
names to the model's, and predict every match on a date with a single model fit.
"""
from __future__ import annotations
import argparse, json, re, urllib.request
import pandas as pd

from .data import load_internationals, load_worldcup, WC_URL
from .ratings import fit
from .halves import fit_half_model
from .venues import lookup as venue_lookup
from .predict import summarize, render
from .rosters import _norm

HOST_NATIONS = {"United States", "Mexico", "Canada"}
_PLACEHOLDER = re.compile(r"^(\d|[WL]\d+$)|/")  # group/knockout slot codes


def is_real_team(name: str) -> bool:
    return bool(name) and not _PLACEHOLDER.search(name)


def load_full_schedule() -> pd.DataFrame:
    raw = urllib.request.urlopen(WC_URL.format(year=2026), timeout=45).read()
    rows = []
    for m in json.loads(raw).get("matches", []):
        rows.append({"date": m.get("date"), "round": m.get("round"),
                     "team1": _norm(m.get("team1")), "team2": _norm(m.get("team2")),
                     "ground": m.get("ground"),
                     "played": bool((m.get("score") or {}).get("ft"))})
    return pd.DataFrame(rows)


def all_teams_2026() -> list:
    """The real qualified teams (model-normalised names)."""
    sched = load_full_schedule()
    teams = set()
    for _, r in sched.iterrows():
        for t in (r["team1"], r["team2"]):
            if is_real_team(t):
                teams.add(t)
    return sorted(teams)


def _host_of(team1, team2, ground):
    """Return the team with home advantage, if a host nation plays in its country."""
    country = (venue_lookup(ground) or [None])[0]
    code = {"USA": "United States", "MEX": "Mexico", "CAN": "Canada"}.get(country)
    if code == team1:
        return team1
    if code == team2:
        return team2
    return None


def predict_date(date: str, continuity=None):
    intl = load_internationals()
    wc = load_worldcup()
    model = fit(intl, intl["date"].max(), continuity=continuity)
    half = fit_half_model(wc)

    sched = load_full_schedule()
    todays = sched[(sched["date"] == date) &
                   sched["team1"].map(is_real_team) & sched["team2"].map(is_real_team)]
    blocks = []
    for _, m in todays.iterrows():
        t1, t2, ground = m["team1"], m["team2"], m["ground"]
        host = _host_of(t1, t2, ground)
        if host == t2:
            l2, l1 = model.expected_goals(t2, t1, neutral=False)
        else:
            l1, l2 = model.expected_goals(t1, t2, neutral=(host != t1))
        splits = half.split(l1, l2)
        res = {ph: summarize(model, splits[ph][0], splits[ph][1], ph) for ph in ("1H", "2H", "FT")}
        note = []
        if t1 not in model.teams or t2 not in model.teams:
            unknown = [t for t in (t1, t2) if t not in model.teams]
            note.append(f"limited data for {', '.join(unknown)} (pooled strength)")
        blocks.append((t1, t2, host, host is None, res, ground, m["round"], note))
    return blocks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--rosters", default=None)
    args = ap.parse_args()
    cont = None
    if args.rosters:
        from .rosters import continuity_scores, load_current_rosters
        cont = continuity_scores(load_current_rosters(args.rosters))
    blocks = predict_date(args.date, continuity=cont)
    print(f"\n{len(blocks)} match(es) on {args.date}")
    for t1, t2, host, neutral, res, ground, rnd, note in blocks:
        print(render(t1, t2, host, neutral, res))
        print(f"   venue: {ground}  |  {rnd}" + (f"  |  NOTE: {'; '.join(note)}" if note else ""))
