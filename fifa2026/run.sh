#!/usr/bin/env bash
# Convenience runner. All scripts cache data under ./cache after first run.
set -e
cd "$(dirname "$0")"
echo "### 1. Variable significance"; python3 -m src.selection
echo; echo "### 2. Cooling-break spike analysis"; python3 -m src.cooling_breaks
echo; echo "### 3. Out-of-sample backtest"; python3 -m src.backtest
echo; echo "### 4. Example prediction"; python3 -m src.predict --team1 Argentina --team2 France
