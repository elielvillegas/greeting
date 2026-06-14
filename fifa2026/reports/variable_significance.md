# Variable significance selection

Keep iff a statistically significant effect is demonstrated (p<0.05).

| Variable | Evidence | p-value | Decision |
|---|---|---|---|
| Team attack/defense strength | LR chi2=3440, df=448 | 0.00e+00 | KEEP |
| Home / host advantage | beta=0.230 (x1.26 goals) | 5.33e-24 | KEEP |
| Half (2H vs 1H scoring) | beta=0.376 (x1.46) | 2.51e-04 | KEEP |
| Stage (knockout vs group) | beta=-0.052 (x0.95) | 0.618 | DROP |
| Venue altitude | no altitude variation in 2014-2026 WC venues | n/a | DROP |
| Venue heat index | per-match weather not reachable (egress) | n/a | DROP |
| Cooling-break post-break effect | see cooling_break_analysis.md (pooled p=1.00) | n/a | DROP |

**Kept:** Team attack/defense strength, Home / host advantage, Half (2H vs 1H scoring)

**Dropped (not demonstrably significant from available data):** Stage (knockout vs group), Venue altitude, Venue heat index, Cooling-break post-break effect

Note: time-decay weighting (recent form) is a fitting choice tuned by out-of-sample likelihood in backtest.py, not a hypothesis test.
