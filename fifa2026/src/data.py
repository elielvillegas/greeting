"""Data acquisition + caching.

Network egress in this environment is GitHub-only, so every source below is a
raw.githubusercontent.com URL. Two sources:

1. martj42/international_results  -> every international match 1872..present
   (full-time scores, venue, neutral flag). Drives team-strength ratings.
2. openfootball/worldcup.json     -> World Cup matches 2014/18/22/26 with
   per-goal MINUTE stamps + HT/FT scores + venue. Drives per-half timing and
   the cooling-break analysis.

Everything is cached under fifa2026/cache so re-runs are offline-fast.
"""
from __future__ import annotations
import io, json, os, urllib.request
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE, exist_ok=True)

INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
WC_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json"
WC_YEARS = [2014, 2018, 2022, 2026]


def _fetch(url: str, fname: str, refresh: bool = False) -> bytes:
    path = os.path.join(CACHE, fname)
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return f.read()
    raw = urllib.request.urlopen(url, timeout=60).read()
    with open(path, "wb") as f:
        f.write(raw)
    return raw


def load_internationals(refresh: bool = False) -> pd.DataFrame:
    """All international results as a tidy DataFrame."""
    raw = _fetch(INTL_URL, "international_results.csv", refresh)
    df = pd.read_csv(io.BytesIO(raw))
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    for c in ("home_score", "away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def _minute(g: dict) -> float:
    """Goal minute as a float; openfootball uses 'minute' + optional 'offset'
    for stoppage time (e.g. 90+1 -> minute=90, offset=1)."""
    try:
        m = float(str(g.get("minute")).strip())
    except (TypeError, ValueError):
        return float("nan")
    return m + float(g.get("offset", 0) or 0)


def load_worldcup(refresh: bool = False) -> pd.DataFrame:
    """One row per *played* World Cup match, with goal-minute lists."""
    rows = []
    for yr in WC_YEARS:
        raw = _fetch(WC_URL.format(year=yr), f"worldcup_{yr}.json", refresh)
        data = json.loads(raw)
        for m in data.get("matches", []):
            score = m.get("score") or {}
            ft = score.get("ft")
            if not ft:  # not yet played
                continue
            ht = score.get("ht") or [None, None]
            rows.append({
                "year": yr,
                "date": pd.to_datetime(m.get("date")),
                "round": m.get("round"),
                "stage": "group" if m.get("group") else "knockout",
                "team1": m.get("team1"),
                "team2": m.get("team2"),
                "ground": m.get("ground"),
                "ft1": ft[0], "ft2": ft[1],
                "ht1": ht[0], "ht2": ht[1],
                "goals1_min": sorted(_minute(g) for g in m.get("goals1", [])),
                "goals2_min": sorted(_minute(g) for g in m.get("goals2", [])),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    intl = load_internationals()
    wc = load_worldcup()
    print(f"internationals: {len(intl):,} matches  {intl.date.min().date()}..{intl.date.max().date()}")
    print(f"world cup matches (played): {len(wc)}")
    print(wc.groupby('year').size().to_string())
    print("\nsample WC row:")
    print(wc.iloc[-1][["year", "date", "team1", "team2", "ft1", "ft2", "ht1", "ht2", "goals1_min", "goals2_min", "ground"]].to_string())
