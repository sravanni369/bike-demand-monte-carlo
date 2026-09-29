"""Does assuming bell-shaped forecast errors under-plan busy days?

A regression forecasts tomorrow's rentals. Capacity = forecast + safety allowance, where the
allowance is either the normal-theory 80th percentile of recent errors (mean + 0.8416 * sd)
or their empirical 80th percentile. Both are scored with the same assumed 4:1 penalty used
in monte_carlo.py. 0.8 is the critical fractile of a 4:1 shortage-to-idle loss.

Everything is one-step-ahead and in date order: the model for day t is fitted only on days
before t, and the allowance for day t uses only errors observed before t. 2012 was already
examined by monte_carlo.py, so every 2012 number here is retrospective.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.dates
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
OUT = ROOT / "results"
SHORT, IDLE = 4.0, 1.0             # assumed cost per rental short / per unit idle
FRACTILE = SHORT / (SHORT + IDLE)  # 0.8
WINDOW = 90                        # past out-of-sample errors used for the allowance
Z = stats.norm.ppf(FRACTILE)
SANDY = "2012-10-29"               # Hurricane Sandy, cnt = 22


def load():
    d = pd.read_csv(ROOT / "data" / "day.csv", parse_dates=["dteday"])
    assert (d.casual + d.registered == d.cnt).all()
    return d


def design(d):
    """Features known the evening before day t: lagged demand and the calendar.
    workingday is left out: it equals weekday Mon-Fri minus holiday, so it adds nothing."""
    X = pd.DataFrame({"const": 1.0,
                      "lag1": d.cnt.shift(1), "lag7": d.cnt.shift(7),
                      "holiday": d.holiday,
                      "doy_sin": np.sin(2 * np.pi * d.dteday.dt.dayofyear / 365.25),
                      "doy_cos": np.cos(2 * np.pi * d.dteday.dt.dayofyear / 365.25)})
    for k in range(1, 7):
        X[f"wd{k}"] = (d.weekday == k).astype(float)
    return X


def one_step_forecasts(d, first=28):
    """Expanding-window OLS refit every day; returns out-of-sample forecasts."""
    X, y = design(d).to_numpy(float), d.cnt.to_numpy(float)
    yhat = np.full(len(d), np.nan)
    for t in range(first, len(d)):
        rows = np.arange(7, t)                       # lag7 exists from index 7
        beta, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
        yhat[t] = X[t] @ beta
    return yhat


def penalty(demand, cap):
    short = np.maximum(demand - cap, 0)
    idle = np.maximum(cap - demand, 0)
    return SHORT * short + IDLE * idle, short, idle


def ljung_box(e, lags=7):
    e = e - e.mean(); n = len(e); denom = np.sum(e**2)
    r = np.array([np.sum(e[k:] * e[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lags + 1)))
    return r, q, stats.chi2.sf(q, lags)


def diagnostics(e):
    r, q, p = ljung_box(e)
    return {"n": int(len(e)), "mean": float(e.mean()), "sd": float(e.std(ddof=1)),
            "skew": float(stats.skew(e)), "kurtosis": float(stats.kurtosis(e, fisher=False)),
            "jarque_bera": float(stats.jarque_bera(e).statistic),
            "jarque_bera_p": float(stats.jarque_bera(e).pvalue),
            "shapiro_p": float(stats.shapiro(e).pvalue),
            "lag1_autocorr": float(r[0]), "ljung_box_q7": float(q), "ljung_box_p": float(p),
            "durbin_watson": float(np.sum(np.diff(e)**2) / np.sum((e - e.mean())**2))}


def block_bootstrap_ci(x, block=7, reps=10_000, seed=0):
    """95% CI of the mean with a moving-block bootstrap (keeps weekly dependence)."""
    rng = np.random.default_rng(seed); n = len(x); k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])


def build_plans(d, idx, drop_err_dates=()):
    err = d.err.copy(); nerr = d.naive_err.copy()
    drop = d.dteday.isin(pd.to_datetime(list(drop_err_dates)))
    err[drop] = np.nan; nerr[drop] = np.nan
    plans = {k: np.empty(len(idx)) for k in
             ["no allowance", "normal allowance", "empirical allowance",
              "naive + normal", "naive + empirical", "raw 90-day quantile", "raw 7-day quantile"]}
    for j, t in enumerate(idx):
        pe = err.iloc[t - WINDOW:t].dropna().to_numpy()
        pn = nerr.iloc[t - WINDOW:t].dropna().to_numpy()
        f, prev = d.yhat.iloc[t], d.cnt.iloc[t - 1]
        plans["no allowance"][j] = f
        plans["normal allowance"][j] = f + pe.mean() + Z * pe.std(ddof=1)
        plans["empirical allowance"][j] = f + np.quantile(pe, FRACTILE)
        plans["naive + normal"][j] = prev + pn.mean() + Z * pn.std(ddof=1)
        plans["naive + empirical"][j] = prev + np.quantile(pn, FRACTILE)
        plans["raw 90-day quantile"][j] = np.quantile(d.cnt.iloc[t - WINDOW:t], FRACTILE)
        plans["raw 7-day quantile"][j] = np.quantile(d.cnt.iloc[t - 7:t], FRACTILE)
    return plans


def main():
    d = load()
    d["yhat"] = one_step_forecasts(d)
    d["err"] = d.cnt - d.yhat
    d["naive_err"] = d.cnt - d.cnt.shift(1)      # yesterday's count as the forecast
    test = d.dteday.dt.year == 2012
    idx = np.flatnonzero(test)
    assert not d.err.iloc[idx[0] - WINDOW:].isna().any()

    plans = build_plans(d, idx)
    y = d.cnt.to_numpy(float)[idx]
    q = d.dteday.iloc[idx].dt.quarter.to_numpy()
    busy_hindsight = y >= np.quantile(y, 0.8)                       # reporting only
    busy_trailing = np.array([d.cnt.iloc[t] >= d.cnt.iloc[t - WINDOW:t].quantile(0.8)
                              for t in idx])                        # known in advance
    pen = {k: penalty(y, c)[0] for k, c in plans.items()}
    rows, by_q = [], []
    for name, cap in plans.items():
        p, sh, idl = penalty(y, cap)
        rows.append({"plan": name, "mean_penalty": p.mean(), "coverage": np.mean(y <= cap),
                     "mean_short": sh.mean(), "mean_idle": idl.mean(),
                     "busy_cov_2012_q80": np.mean((y <= cap)[busy_hindsight]),
                     "busy_cov_trailing_q80": np.mean((y <= cap)[busy_trailing])})
        for k in range(1, 5):
            by_q.append({"plan": name, "quarter": f"Q{k}", "mean_penalty": p[q == k].mean()})
    table = pd.DataFrame(rows).set_index("plan")
    qtab = pd.DataFrame(by_q).pivot(index="plan", columns="quarter", values="mean_penalty")

    gaps = {}
    for a, b in [("normal allowance", "empirical allowance"),
                 ("naive + empirical", "empirical allowance"),
                 ("raw 90-day quantile", "empirical allowance"),
                 ("raw 7-day quantile", "empirical allowance")]:
        x = pen[a] - pen[b]
        gaps[f"{a} minus {b}"] = {"mean": float(x.mean()),
                                  "ci95_block7": [float(v) for v in block_bootstrap_ci(x)]}

    no_sandy = build_plans(d, idx, drop_err_dates=[SANDY])
    sandy_gap = (penalty(y, no_sandy["normal allowance"])[0].mean()
                 - penalty(y, no_sandy["empirical allowance"])[0].mean())

    naive_mae = np.mean(np.abs(y - d.cnt.to_numpy(float)[idx - 1]))
    reg_mae = np.mean(np.abs(d.err.to_numpy()[idx]))
    mae_diff = np.abs(d.err.to_numpy()[idx]) - np.abs(y - d.cnt.to_numpy(float)[idx - 1])

    e11 = d.err[(~test) & d.err.notna()].to_numpy()
    e11_late = d.err[(~test) & (d.index >= 90)].to_numpy()
    worst3 = np.argsort(np.abs(e11))[-3:]
    diag = {"2011 from day 29": diagnostics(e11),
            "2011 from day 90": diagnostics(e11_late),
            "2011 minus 3 largest": diagnostics(np.delete(e11, worst3)),
            "2012": diagnostics(d.err[test].to_numpy()),
            "2012 minus Sandy + next day": diagnostics(
                d.err[test & ~d.dteday.isin(pd.to_datetime([SANDY, "2012-10-30"]))].to_numpy())}

    # leakage check: corrupt every day from a cutoff; nothing before it may change
    cut = int(idx[180]); d2 = d.copy(); d2.loc[cut:, "cnt"] += 10_000
    d2["yhat"] = one_step_forecasts(d2); d2["err"] = d2.cnt - d2.yhat
    d2["naive_err"] = d2.cnt - d2.cnt.shift(1)
    p2 = build_plans(d2, idx)
    before = idx <= cut
    moved = max(np.nanmax(np.abs(d2.yhat.to_numpy()[:cut + 1] - d.yhat.to_numpy()[:cut + 1])),
                max(np.max(np.abs(p2[k][before] - plans[k][before])) for k in plans))

    pd.set_option("display.width", 140)
    print(f"Test: {len(idx)} days of 2012 (retrospective), one-step ahead. Loss 4:1, "
          f"fractile {FRACTILE}, z = {Z:.4f}, allowance window {WINDOW} errors.")
    print(f"Point forecast MAE: regression {reg_mae:,.1f} vs yesterday's count {naive_mae:,.1f}; "
          f"difference {mae_diff.mean():+.1f}, block-7 95% CI "
          f"[{block_bootstrap_ci(mae_diff)[0]:.1f}, {block_bootstrap_ci(mae_diff)[1]:.1f}]")
    print("\nError shape (one-step regression errors)")
    for label, dg in diag.items():
        print(f"  {label:28s} n={dg['n']:3d} skew={dg['skew']:+.2f} kurt={dg['kurtosis']:.2f} "
              f"JB p={dg['jarque_bera_p']:.2g} Shapiro p={dg['shapiro_p']:.2g} "
              f"lag1 r={dg['lag1_autocorr']:+.2f} Ljung-Box(7) p={dg['ljung_box_p']:.2g}")
    print("\nCapacity plans, 2012")
    print(table.round(3).to_string())
    print("\nMean penalty by quarter")
    print(qtab.round(1).to_string())
    print("\nPaired gaps in mean penalty per day (positive = second plan cheaper)")
    for k, v in gaps.items():
        print(f"  {k:48s} {v['mean']:8.1f}  block-7 95% CI "
              f"[{v['ci95_block7'][0]:.1f}, {v['ci95_block7'][1]:.1f}]")
    print(f"  normal minus empirical with Sandy's error kept out of the windows: {sandy_gap:.1f}")
    print(f"\nLeakage check: max change in any forecast or plan before the cutoff = {moved:.6f}")
    assert moved == 0.0

    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "error_allowance_plans.csv")
    qtab.to_csv(OUT / "error_allowance_by_quarter.csv")
    json.dump({"diagnostics": diag, "gaps": gaps, "sandy_excluded_gap": sandy_gap,
               "regression_mae": reg_mae, "naive_mae": naive_mae,
               "leakage_max_change": moved}, open(OUT / "error_allowance_metrics.json", "w"),
              indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(e11, bins=30, density=True, color="#8aa")
    xs = np.linspace(e11.min(), e11.max(), 200)
    ax[0].plot(xs, stats.norm.pdf(xs, e11.mean(), e11.std(ddof=1)), "k-")
    ax[0].set_title("2011 one-step errors vs normal curve")
    stats.probplot(e11, dist="norm", plot=ax[1]); ax[1].set_title("Normal Q-Q, 2011 errors")
    ax[2].plot(d.dteday, d.err, lw=0.7); ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_title("One-step forecast error over time")
    ax[2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax[2].xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    fig.tight_layout(); fig.savefig(OUT / "error_diagnostics.png", dpi=120)


if __name__ == "__main__":
    main()
