"""Deep Learning, section 17.1.2: Monte Carlo workload planning (50 lines)."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

def loss(capacity, demand):
    return 4 * np.maximum(demand - capacity, 0) + np.maximum(capacity - demand, 0)

def evaluate(frame, seed=42, draws=5000):
    if draws < 1 or frame.empty:
        raise ValueError("Positive draws and nonempty data required")
    if frame.dteday.duplicated().any() or frame.cnt.isna().any() or (frame.cnt < 0).any():
        raise ValueError("Invalid dates or demand")
    frame = frame.sort_values("dteday").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    grid = np.arange(0, 10001, 250)[:, None]
    rows = []
    for i in range(90, len(frame)):
        today = frame.iloc[i]
        if today.dteday.year != 2012:
            continue
        history = frame.iloc[i - 90:i].cnt.to_numpy()
        simulated = rng.choice(history, size=draws, replace=True)
        estimated = loss(grid, simulated).mean(axis=1)
        exact = loss(grid, history).mean(axis=1)
        mc = int(grid[np.argmin(estimated), 0])
        reference = int(grid[np.argmin(exact), 0])
        baseline = int(grid[np.argmin(abs(grid[:, 0] - history.mean())), 0])
        rows.append([str(today.dteday.date()), int(today.cnt), baseline, mc, reference,
                     float(exact[np.argmin(estimated)] - exact.min())])
    result = pd.DataFrame(rows, columns=["date", "demand", "baseline", "mc", "exact", "regret"])
    if result.empty:
        raise ValueError("No 2012 evaluation days")
    metrics = {"seed": seed, "draws": draws, "test_days": len(result)}
    for name in ["baseline", "mc", "exact"]:
        metrics[name + "_loss"] = round(float(loss(result[name], result.demand).mean()), 3)
        metrics[name + "_coverage"] = round(float((result[name] >= result.demand).mean()), 4)
    metrics["mc_regret"] = round(float(result.regret.mean()), 4)
    metrics["loss_reduction_pct"] = round(100 * (1 - metrics["mc_loss"] / metrics["baseline_loss"]), 2) if metrics["baseline_loss"] else None
    return result, metrics

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    frame = pd.read_csv(base / "data/day.csv", parse_dates=["dteday"])
    result, metrics = evaluate(frame, int(sys.argv[1]) if len(sys.argv) > 1 else 42)
    (base / "results").mkdir(exist_ok=True)
    result.to_csv(base / "results/backtest.csv", index=False)
    print(json.dumps(metrics, indent=2))
