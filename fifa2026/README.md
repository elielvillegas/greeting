# FIFA World Cup 2026 — per-match prediction model

A statistical model that predicts **scoring, result and the winner** of a match
for **1st half, 2nd half and full game**, plus a dedicated empirical study of
whether anything measurably changes after a **cooling break**.

## Approach (short version)
- **Engine:** Dixon–Coles / Maher **Poisson regression**. Each team has latent
  *attack* and *defense* strengths; expected goals depend on attack vs opponent
  defense + home advantage, with a low-score correction (ρ) and exponential
  **time-decay** weighting (recent form counts more). Fit with `statsmodels`, so
  every effect comes with a p-value.
- **From one fit we get all three asks:** the full scoreline distribution yields
  expected goals (*scoring*), summed cells give W/D/L (*result*), and the larger
  win probability gives the *winner*.
- **Halves:** full-time expected goals are split into 1H/2H using the empirical
  goal share measured from World Cup HT/FT scores (~41% / 59%).

## Only statistically significant variables (`src/selection.py`)
Each candidate is tested; only demonstrated effects (p<0.05) enter the model.

| Kept (significant) | Dropped (not demonstrable from available data) |
|---|---|
| Team attack/defense (LR p≈0) | Stage knockout vs group (p=0.62) |
| Home/host advantage (p=1e-27) | Venue altitude / heat (no variation / weather unreachable) |
| Half 2H vs 1H (p=2.5e-4) | Cooling-break effect (pooled p=1.00) |

## Cooling breaks (`src/cooling_breaks.py`)
Your question — *does anything spike after the break?* — tested on goal-minute
data (2014/18/22 at ~30'/75'; 2026 mandatory at ~22'/67'). **Result: no
statistically significant change in goal rate or momentum after the break.** The
perceived rhythm change does not translate into measurable goals, so per the
"significant variables only" rule it is excluded from the predictor.
(Possession/xG are not reachable under this environment's GitHub-only network,
so only goal events are tested — see report header.)

## Validation (`src/backtest.py`)
Train strictly before a tournament, predict it, score with log-loss / Brier /
**RPS**. Beats the uniform baseline on Brier+RPS for 2018 and 2022; 2022 log-loss
suffers from that tournament's historic upsets.

## Data (GitHub-only egress)
- `martj42/international_results` — all internationals 1872→present (strengths/form)
- `openfootball/worldcup.json` — WC 2014/18/22/26 with goal minutes + HT/FT + venue

## Run it
```bash
pip install -r requirements.txt
python -m src.predict --team1 Argentina --team2 Mexico        # neutral
python -m src.predict --team1 Mexico --team2 "United States" --host Mexico
python -m src.cooling_breaks      # the break analysis
python -m src.selection           # significance table
python -m src.backtest            # validation
./run.sh                          # all of the above
```
Data is cached under `cache/` after first download.

## Honest limitations
- Per-match winner only (no full-bracket Monte Carlo), by design.
- No possession/xG (network-restricted); cooling-break study is goal-events only.
- 2026 sample is tiny and grows as the tournament proceeds (re-run to refresh).
