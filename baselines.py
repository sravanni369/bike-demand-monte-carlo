"""Audit-driven baselines and stability checks; outside the 50-line core.

Answers the questions a sceptical reader asks of monte_carlo.py:
  1. Is the gain from Monte Carlo sampling, or from planning for the asymmetric loss?
     (For a 4:1 shortage:idle loss the closed-form optimum is the 0.8 quantile of the window.)
  2. Does the whole-year 40.56% hold in every quarter?
  3. Does the 90-day window matter more than the method? (Post-hoc sensitivity, not a tuned result.)
  4. What is the coverage compared with, and how wide is the interval on the reduction?
Run:  python baselines.py   (after python monte_carlo.py has written results/backtest.csv)
"""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
from monte_carlo import loss

base = Path(__file__).resolve().parent
frame = pd.read_csv(base / "data/day.csv", parse_dates=["dteday"]).sort_values("dteday").reset_index(drop=True)
rows = pd.read_csv(base / "results/backtest.csv", parse_dates=["date"])
grid = np.arange(0, 10001, 250)
cnt = frame.cnt.to_numpy()
idx = [i for i in range(90, len(frame)) if frame.dteday[i].year == 2012]
demand = cnt[idx]
assert np.array_equal(demand, rows.demand.to_numpy())
snap = lambda x: int(grid[np.argmin(abs(grid - x))])
plan = lambda fn, w=90: np.array([fn(cnt[i - w:i]) for i in idx])
base_loss = loss(rows.baseline.to_numpy(), demand).mean()
out = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "plans": {}}


def report(name, cap, tag):
    L = loss(cap, demand)
    cov = float((cap >= demand).mean())
    red = 100 * (1 - L.mean() / base_loss)
    print(f"  {name:52s} penalty {L.mean():9.3f}   covered {100 * cov:5.1f}%   vs mean {red:6.1f}%")
    out["plans"][tag] = {"penalty": round(float(L.mean()), 3), "coverage": round(cov, 4), "reduction_pct": round(float(red), 2)}
    return L


print("[1] same 90-day window, different decision rule (test days: 2012, n=366)")
report("previous-90-day mean, on grid (README baseline)", rows.baseline.to_numpy(), "mean90")
report("Monte Carlo, 5,000 draws, seed 42 (README)", rows.mc.to_numpy(), "mc90")
report("exact expectation over the 90 days (README)", rows.exact.to_numpy(), "exact90")
report("0.8 quantile of the 90 days: np.quantile(h, 0.8)", plan(lambda h: np.quantile(h, 0.8)), "q80_90")
report("mean + 1 std of the 90 days, on grid", plan(lambda h: snap(h.mean() + h.std())), "mean_std90")

print("\n[2] same 0.8-quantile rule, shorter windows (post-hoc sensitivity, not tuned)")
for w in (7, 14, 30, 60):
    report(f"0.8 quantile of the previous {w} days", plan(lambda h: np.quantile(h, 0.8), w), f"q80_{w}")
report("max of the previous 7 days", plan(lambda h: h.max(), 7), "max7")
report("perfect foresight (lower bound, for scale)", demand, "oracle")

print("\n[3] the 40.56% by quarter (Monte Carlo vs previous-90-day mean)")
d = rows.copy()
d["Lb"] = loss(d.baseline, d.demand)
d["Lm"] = loss(d.mc, d.demand)
d["cov"] = d.mc >= d.demand
q = d.groupby(d.date.dt.quarter).agg(days=("demand", "size"), demand=("demand", "mean"), mean_plan=("Lb", "mean"), mc_plan=("Lm", "mean"), covered=("cov", "mean"))
q["reduction_pct"] = 100 * (1 - q.mc_plan / q.mean_plan)
print(q.round(2).to_string())
out["by_quarter"] = {int(k): {c: round(float(v[c]), 2) for c in q.columns} for k, v in q.iterrows()}
print(f"  days MC plan beats mean plan: {int((d.Lm < d.Lb).sum())} / 366")

rng = np.random.default_rng(0)
blocks = [np.arange(i, min(i + 7, len(d))) for i in range(0, len(d), 7)]
B = []
for _ in range(5000):
    s = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
    B.append(100 * (1 - d.Lm.values[s].mean() / d.Lb.values[s].mean()))
lo, hi = np.percentile(B, [2.5, 97.5])
print(f"  weekly block bootstrap, 5,000 resamples, 95% interval on the reduction: {lo:.1f}% to {hi:.1f}%")
out["reduction_block_bootstrap_95"] = [round(float(lo), 1), round(float(hi), 1)]

print("\n[4] what the coverage should be compared with")
in_window = np.mean([(cnt[i - 90:i] <= rows.mc[k]).mean() for k, i in enumerate(idx)])
above_mean = (demand > plan(lambda h: h.mean())).mean()
print(f"  target implied by the 4:1 loss             : 80.0% of days")
print(f"  MC plan, coverage inside its own window     : {100 * in_window:.1f}%")
print(f"  MC plan, coverage on the 2012 test days     : {100 * (rows.mc >= rows.demand).mean():.2f}%")
print(f"  2012 days with demand above the 90-day mean : {100 * above_mean:.1f}%  (the series is rising)")
out["coverage"] = {"target": 0.8, "in_window": round(float(in_window), 4), "test": round(float((rows.mc >= rows.demand).mean()), 4), "demand_above_mean90": round(float(above_mean), 4)}
(base / "results/baselines.json").write_text(json.dumps(out, indent=2) + "\n")
