"""Host-city context for World Cup 2026.

Two physical conditions that VARY across the 16 host cities and are candidate
predictors: altitude (acclimatization / ball flight) and typical June-July heat.
The cooling break itself is mandatory in every 2026 match, so it cannot
discriminate between matches -- but the heat that motivates it does vary.

Values are approximate typical match-time conditions, clearly an estimate.
`roofed` stadiums are climate-controllable, which damps the heat effect.
"""
from __future__ import annotations

# city substring -> (country, altitude_m, summer_heat_index_c, roofed)
HOST_CITIES = {
    # Mexico -- altitude is the headline factor
    "Mexico City":   ("MEX", 2240, 26, False),
    "Guadalajara":   ("MEX", 1566, 30, False),
    "Monterrey":     ("MEX",  540, 35, False),
    # USA
    "Atlanta":       ("USA",  320, 33, True),
    "Boston":        ("USA",   40, 27, False),
    "Foxborough":    ("USA",   40, 27, False),
    "Dallas":        ("USA",  130, 36, True),
    "Arlington":     ("USA",  130, 36, True),
    "Houston":       ("USA",   30, 37, True),
    "Kansas City":   ("USA",  280, 34, False),
    "Los Angeles":   ("USA",   40, 28, True),
    "Inglewood":     ("USA",   40, 28, True),
    "Miami":         ("USA",    2, 35, False),
    "New York":      ("USA",    5, 29, False),
    "New Jersey":    ("USA",    5, 29, False),
    "East Rutherford":("USA",   5, 29, False),
    "Philadelphia":  ("USA",   12, 31, False),
    "San Francisco": ("USA",    9, 24, False),
    "Santa Clara":   ("USA",    9, 24, False),
    "Seattle":       ("USA",   50, 24, False),
    # Canada
    "Toronto":       ("CAN",   76, 26, False),
    "Vancouver":     ("CAN",    0, 22, True),
}

# Reference baselines used to center the covariates in the model.
ALT_BASELINE_M = 100.0     # near sea level
HEAT_BASELINE_C = 25.0     # mild


def lookup(ground: str | None):
    """Return (country, altitude_m, heat_c, roofed) for a ground string, or a
    neutral sea-level/mild default if unknown (e.g. non-2026 venues)."""
    if ground:
        for key, val in HOST_CITIES.items():
            if key.lower() in ground.lower():
                return val
    return (None, int(ALT_BASELINE_M), int(HEAT_BASELINE_C), False)


def venue_features(ground: str | None) -> dict:
    """Centered covariates for the scoring model. Roofed venues get their heat
    pulled toward the baseline (climate control)."""
    country, alt, heat, roofed = lookup(ground)
    eff_heat = HEAT_BASELINE_C + 0.4 * (heat - HEAT_BASELINE_C) if roofed else heat
    return {
        "host_country": country,
        "altitude_km": (alt - ALT_BASELINE_M) / 1000.0,   # per-km above baseline
        "heat_excess": (eff_heat - HEAT_BASELINE_C) / 10.0,  # per-10C above baseline
        "roofed": roofed,
    }


if __name__ == "__main__":
    for g in ["Mexico City", "Houston", "Vancouver", "Doha", None]:
        print(f"{str(g):14} -> {venue_features(g)}")
