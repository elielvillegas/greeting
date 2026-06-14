"""Roster-continuity hook.

Idea (your request): if a team's squad has turned over since the last World Cup,
its older matches describe a different team and should count less. We compute a
per-team continuity score in [0,1] and use it to discount that team's PRE-2026
matches (wired into ratings.build_long via the `continuity` argument).

  continuity[team] = Jaccard(current 2026 squad, 2022 squad)
                   = |players in both| / |players in either|

2022 squads come from StatsBomb open-data (full lineups, on GitHub). Current 2026
squads are NOT reachable here, so you supply them: generate a template, fill it,
pass it to the predictor.

  python -m src.rosters --template      # writes cache/rosters_2026.json to edit
  python -m src.rosters --show          # prints continuity from the filled file
"""
from __future__ import annotations
import argparse, json, os, urllib.request
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")
SB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
WC2022 = (43, 106)  # competition_id, season_id
SQUADS_2022 = os.path.join(CACHE, "squads_2022.json")
ROSTERS_2026 = os.path.join(CACHE, "rosters_2026.json")

# Aliases that reconcile StatsBomb / openfootball names with the model
# (martj42) names, so continuity keys line up with the teams in the GLM.
TEAM_ALIASES = {
    "IR Iran": "Iran", "Korea Republic": "South Korea",
    "USA": "United States",                 # openfootball 2026 -> martj42
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def _get(url):
    return json.load(urllib.request.urlopen(url, timeout=45))


def _norm(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def get_2022_squads(refresh: bool = False) -> dict:
    """{team: [player names]} for WC2022, unioned across all its matches."""
    if os.path.exists(SQUADS_2022) and not refresh:
        with open(SQUADS_2022) as f:
            return json.load(f)
    comp, season = WC2022
    matches = _get(f"{SB}/matches/{comp}/{season}.json")
    squads: dict[str, set] = {}
    for m in matches:
        lu = _get(f"{SB}/lineups/{m['match_id']}.json")
        for team in lu:
            t = _norm(team["team_name"])
            squads.setdefault(t, set())
            for p in team["lineup"]:
                squads[t].add(p["player_name"])
    out = {t: sorted(v) for t, v in squads.items()}
    with open(SQUADS_2022, "w") as f:
        json.dump(out, f, indent=1)
    return out


def teams_2026() -> list:
    from .data import load_worldcup
    wc = load_worldcup()
    g = wc[wc["year"] == 2026]
    return sorted({_norm(t) for t in set(g["team1"]) | set(g["team2"])})


def write_template(path: str = ROSTERS_2026):
    """Write an editable {team: [players]} file for ALL 48 qualified teams,
    pre-filled with each team's 2022 squad as a starting point. Teams that did
    not play in 2022 (World Cup newcomers) are left empty and get NO roster-
    continuity adjustment -- they are treated purely as new teams."""
    from .fixtures import all_teams_2026
    sq = get_2022_squads()
    teams = all_teams_2026()
    template = {t: sq.get(t, []) for t in teams}
    newcomers = [t for t in teams if t not in sq]
    with open(path, "w") as f:
        json.dump(template, f, indent=1, ensure_ascii=False)
    return path, len(template), newcomers


def load_current_rosters(path: str = ROSTERS_2026) -> dict:
    with open(path) as f:
        return json.load(f)


def _jaccard(a, b) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def continuity_scores(current: dict | None = None) -> dict:
    """{team: continuity in [0,1]} vs the 2022 squad. Teams absent from either
    side are omitted -> the model treats them as continuity 1.0 (no discount)."""
    if current is None:
        current = load_current_rosters()
    hist = get_2022_squads()
    return {t: _jaccard(current[t], hist[t]) for t in current if t in hist and current[t]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true", help="write editable roster template")
    ap.add_argument("--show", action="store_true", help="print continuity from filled file")
    args = ap.parse_args()
    if args.template:
        p, n, newcomers = write_template()
        print(f"wrote template for {n} teams -> {p}")
        print(f"{len(newcomers)} World Cup newcomers (no 2022 squad, treated as new -> no continuity adjustment):")
        print("  " + ", ".join(newcomers))
        print("Edit returning teams (drop departed players, add new caps), then use --show or pass to predict.")
    if args.show:
        scores = continuity_scores()
        print("Returning teams (continuity vs 2022 squad):")
        for t, c in sorted(scores.items(), key=lambda kv: kv[1]):
            print(f"  {t:24} continuity={c:.2f}")
