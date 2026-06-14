# Cooling-break spike analysis

Window = +/-8 min around each break. Test: post-break goal count ~ Binomial(pre+post, 0.5) under no-change null.


## 2026 (mandatory breaks ~22'/67', n=8 matches)
- break 22': pre=3 post=2 rate_ratio=0.67 p=1.000
- break 67': pre=2 post=2 rate_ratio=1.00 p=1.000
- **pooled both breaks**: pre=5 post=4 rate_ratio=0.80 p=1.000 -> not significant

## 2014-2022 (heat breaks ~30'/75', n=192 matches)
- break 30': pre=32 post=45 rate_ratio=1.41 p=0.171
- break 75': pre=57 post=44 rate_ratio=0.77 p=0.232
- **pooled both breaks**: pre=89 post=89 rate_ratio=1.00 p=1.000 -> not significant

## Momentum: who scores next after a break
- 2026: next scorer was leading=3, level=4, trailing=4 (trailing vs leading p=1.000)
- 2014-2022: next scorer was leading=60, level=149, trailing=57 (trailing vs leading p=0.853)

## Interpretation
- A rate_ratio > 1 with p < 0.05 means scoring genuinely spikes after the break;
  < 1 means a lull (rhythm disruption). Absent significance, the viewer-felt
  rhythm change does NOT translate into a measurable goal-rate effect, so per
  the 'significant variables only' rule it is excluded from the predictor.
- Possession/xG are unavailable under current egress; this tests goal events only.
