# Empirical distribution on a real bike shop: the quantile did the work, not the draws

**A 50-line Python implementation of the empirical distribution (Eq. 3.28) and Monte Carlo
expectation (Eq. 17.3) from Goodfellow, Bengio and Courville's *Deep Learning*, run on 731
days of real Capital Bikeshare rentals.** Planning tomorrow's rental capacity for an assumed
4:1 shortage-to-idle penalty cut the mean penalty by **40.56%** against a rolling-mean plan
over the 366 days of 2012. Three things the audit added before publishing:

- a one-line `np.quantile(history, 0.8)` matches the 5,000-draw Monte Carlo plan, so the gain
  comes from planning for the asymmetric loss, not from sampling;
- the whole-year figure hides a **negative fourth quarter** (-33.5%), when demand was falling;
- a 7-day window with the same rule scores 59.3%, so the window length matters more than the
  method. That is a post-hoc sensitivity, not a tuned result.

This is a retrospective experiment on public data, not a deployed system or a measured saving.

**Part 2 (2026-09-28)** adds a regression forecast and tests whether its errors are bell-shaped.
They are not, but a bell-curve safety allowance planned as well as an empirical one. See
[Part 2](#part-2-are-the-forecast-errors-bell-shaped-and-does-it-matter-2026-09-28).

Author: Lakshmi Sravani Putta.

## Business question

A bike-rental operator plans daily processing capacity. Under-planning (turned-away rentals,
queues, rebalancing) is assumed to cost four times as much per rental as unused capacity.
How much capacity should be planned for tomorrow, given only the past?

Capacity here means **rental transactions per day**, not bicycles or staff. One bicycle
supports many rentals, so this dataset cannot size a fleet. The objective is
`4 * max(demand - capacity, 0) + max(capacity - demand, 0)`; the 4:1 weights are a declared
assumption, not Capital Bikeshare's costs. Candidate capacities run 0 to 10,000 in steps of 250.
The 90-day lookback, grid and weights were fixed before the test and not tuned on it.

## Book to implementation

Source: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
Local PDF page = printed page + 16.

| Printed page | What the book says | What the code does |
|---|---|---|
| 65, Eq. 3.27 | The Dirac delta puts all mass at one point | Not evaluated numerically; rental counts are discrete, so frequencies suffice (the book says so on p. 66) |
| 66, Eq. 3.28 | The empirical distribution puts mass 1/m on each of m observations | Each of the previous 90 days gets weight 1/90; repeated counts accumulate mass |
| 588, Eq. 17.3 (ch. 17, authors' website) | Estimate an expectation by averaging n samples drawn from the distribution | 5,000 draws with replacement from the 90 days; average the penalty of every candidate capacity; pick the minimum |

**These pages contain equations, not a Python listing.** `monte_carlo.py` is an original
implementation of those two equations, not copied book code. Chapter 17 is not in the local
PDF (which ends inside chapter 8); it is cited from
[deeplearningbook.org](https://www.deeplearningbook.org/contents/monte_carlo.html).

## Dataset

[UCI Bike Sharing](https://doi.org/10.24432/C5W894), Hadi Fanaee-T (2013), CC BY 4.0.
`data/day.csv` holds 731 consecutive daily rows for 2011 to 2012 from Capital Bikeshare,
Washington DC, unmodified. Download URL and SHA-256 are in `data/provenance.json`. Only the
`cnt` column enters a decision; `casual` and `registered` are used to check that `cnt` is their
sum. No weather or calendar feature is used.

## Evaluation

For every day of 2012, use **only the previous 90 days**, choose a capacity, then score it
against that day's actual rentals. A rolling one-step backtest in date order, 366 test days.
`validate.py` perturbs every count from two cutoffs by +10,000 and checks that no decision
before the cutoff changes.

### Same 90-day window, different decision rule

| Plan | Mean penalty / day (lower is better) | Days covered |
|---|---:|---:|
| Previous-90-day mean, on the grid (baseline) | 3,275.407 | 37.98% |
| Monte Carlo, 5,000 draws, seed 42 | 1,947.019 | 64.21% |
| Exact expectation over the same 90 days | 1,936.623 | 64.48% |
| `np.quantile(history, 0.8)`, one line, no grid | 1,945.538 | 65.03% |
| Mean + 1 standard deviation, on the grid | 1,840.407 | 73.0% |

For a 4:1 loss the optimum is the 80th percentile of the window, so the Monte Carlo and exact
plans are both approximations of a one-line quantile. **The 40.56% measures loss-aware versus
loss-unaware planning, not sampling.** Four Monte Carlo seeds (42, 7, 21, 84) give 40.56%,
40.35%, 40.94% and 40.70%: sampling noise is the least interesting spread here.

### The 40.56% by quarter

| Quarter 2012 | Mean demand | Mean plan | Monte Carlo plan | Reduction |
|---|---:|---:|---:|---:|
| Q1 | 4,008 | 4,067.5 | 2,400.2 | +41.0% |
| Q2 | 6,296 | 4,746.3 | 1,763.9 | +62.8% |
| Q3 | 6,920 | 2,404.6 | 1,079.8 | +55.1% |
| Q4 | 5,165 | 1,907.8 | 2,547.3 | **-33.5%** |

The Monte Carlo plan beats the mean plan on 208 of 366 days. When demand falls in Q4, an 80th
percentile of a window drawn from the summer over-provisions and loses to the mean. A weekly
block bootstrap over the test days (5,000 resamples) puts the annual reduction at
**29.8% to 49.5%**.

### Coverage, compared with what it should be compared with

The 4:1 loss implies planning at the 80th percentile. Inside its own 90-day window the plan
covers 80.9% of days. On the 2012 test days it covers 64.21%. That 16-point gap is
nonstationarity: 2012 demand ran above its trailing 90-day mean on 62.3% of days. The series
was rising, which is also why the mean plan lags so badly in Q1 to Q3.

### Window length (post-hoc sensitivity on the same test year, not tuned)

| Same 0.8-quantile rule | Mean penalty / day | Days covered | vs mean plan |
|---|---:|---:|---:|
| Previous 7 days | 1,332.836 | 71.0% | 59.3% |
| Previous 14 days | 1,385.281 | 73.5% | 57.7% |
| Previous 30 days | 1,417.996 | 75.4% | 56.7% |
| Previous 60 days | 1,675.137 | 68.9% | 48.9% |
| Previous 90 days (the project's choice) | 1,945.538 | 65.0% | 40.6% |
| Max of the previous 7 days | 1,343.172 | 85.8% | 59.0% |

These were run after the 90-day result was known, so they are reported as a sensitivity, not
promoted to the headline.

## Part 2: are the forecast errors bell-shaped, and does it matter? (2026-09-28)

**Question.** A regression forecasts tomorrow's rentals and a safety allowance is added on top.
If the allowance assumes the errors follow a normal curve, does that under-plan busy days?

**Answer on 2012: no.** The errors are clearly not normal, but the normal-theory allowance and
the empirical one are statistically indistinguishable, and the normal one covered more busy
days. The errors are skewed to the downside (sudden demand crashes on storm days), which widens
the standard deviation, so the bell-curve allowance over-plans rather than under-plans.

2012 was already examined in Part 1, so everything below is **retrospective**, not a fresh test.

### Method

- **Forecast:** ordinary least squares refitted every day on all earlier days (expanding window),
  one step ahead. Features known the evening before: yesterday's rentals, rentals seven days
  earlier, holiday, weekday, and day of year (sine and cosine). Same-day weather is left out
  because it is not known the night before.
- **Allowance:** from the previous 90 one-step errors, either the normal 80th percentile
  (mean + 0.8416 × sd) or the empirical 80th percentile. 0.8 is the critical fractile of the
  same assumed 4:1 shortage-to-idle penalty as Part 1.
- **Baselines:** no allowance; yesterday's count plus either allowance; the raw 90-day and 7-day
  80th-percentile plans from Part 1.
- **Leakage check:** every count from mid-2012 onward is raised by 10,000; no forecast or plan
  before that date changes (maximum change 0.000000).

### Error shape

| One-step regression errors | n | Skew | Kurtosis | Jarque–Bera p | Ljung–Box(7) p |
|---|---:|---:|---:|---:|---:|
| 2011, from day 29 | 337 | -0.64 | 4.11 | 1.6e-09 | 0.0083 |
| 2011, from day 90 | 275 | -0.68 | 3.97 | 1.1e-07 | 0.063 |
| 2011, three largest removed | 334 | -0.41 | 2.89 | 0.008 | 2.4e-07 |
| 2012 | 366 | -1.11 | 6.98 | 2.5e-69 | 0.00047 |
| 2012, Hurricane Sandy and next day removed | 364 | -0.91 | 6.27 | 5.4e-47 | 6.7e-05 |

The left skew holds under every cut. The heavy tail in 2011 comes from a handful of storm days
(kurtosis 2.89 once the three worst are removed). Evidence of autocorrelation in 2011 depends
on where the series starts (p = 0.0083 from day 29, 0.063 from day 90).
Plots: `results/error_diagnostics.png`.

### Capacity plans, 2012

| Plan | Mean penalty / day | Coverage | Busy-day coverage* |
|---|---:|---:|---:|
| Regression, no allowance | 2,695.555 | 27.0% | 1.4% |
| Regression + normal allowance | 1,357.785 | 83.3% | 73.0% |
| Regression + empirical allowance | 1,336.516 | 77.0% | 60.8% |
| Yesterday's count + normal allowance | 1,671.072 | 84.4% | 78.4% |
| Yesterday's count + empirical allowance | 1,627.255 | 78.1% | 71.6% |
| Raw 90-day 80th percentile (Part 1) | 1,945.538 | 65.0% | 25.7% |
| Raw 7-day 80th percentile (Part 1 sensitivity) | 1,332.836 | 71.0% | 36.5% |

\*Busy day = 2012 demand at or above the 2012 80th percentile. That threshold uses hindsight
and is for reporting only. With a threshold known in advance (the trailing 90-day 80th
percentile), normal vs empirical is 68.0% vs 57.0%, the same direction.

Paired differences in mean penalty per day, with 95% intervals from a weekly block bootstrap:

| Comparison | Gap | 95% CI |
|---|---:|---:|
| Normal minus empirical allowance | 21.3 | [-16.7, 55.4] |
| Yesterday + empirical minus regression + empirical | 290.7 | [171.8, 421.4] |
| Raw 90-day minus regression + empirical | 609.0 | [247.4, 932.3] |
| Raw 7-day minus regression + empirical | -3.7 | [-143.2, 122.9] |

What this shows:

- **Normal vs empirical allowance:** no difference is demonstrated. Part of the small gap is one
  storm: with Hurricane Sandy's error kept out of the windows the gap falls from 21.3 to 15.6.
  The normal allowance over-covers (83.3% against an 80% target) and the empirical one
  under-covers (77.0%).
- **Regression vs yesterday's count:** with the same empirical allowance the regression plan is
  17.9% cheaper, and the interval excludes zero. Its point forecast is not more accurate (MAE
  879.2 vs 870.2, difference +9.0, CI [-43.3, 65.6]), so the gain comes from a steadier error
  distribution for the allowance, not from better point forecasts.
- **Regression vs the Part 1 plans:** it beats the 90-day quantile, but most of that gain is the
  90-day window lagging the 2012 growth. It ties the 7-day quantile, which wins Q2 and Q3. Both
  the regression and the 7-day window were chosen after 2012 had been seen.

### Book exercise (fictitious data)

`book_exercise/hellwig_jarque_bera_book.py` reproduces the Hellwig and Jarque–Bera tests of
Welc, J., & Rodriguez Esquerdo, P. J. (2018). *Applied Regression Analysis for Business:
Tools, Traps and Applications*. Springer, sections 4.3.5–4.3.6, on the book's 20 fictitious
restaurant-cost residuals (Table 4.2). Results match the book: 7 empty Hellwig cells (bounds
4 to 9) and JB = 0.3837 against a critical value of 5.99. The book's worked example uses the
n − 1 standard deviation even though its Table 4.3 writes 1/n; with 1/n the count is 8. Hellwig
is designed for fewer than 30 residuals, so it is not applied to the 366-day bike errors.

## Run and verify

Verified on Windows 11, Python 3.13.5, NumPy 2.2.6, pandas 2.3.1 (the VS Code screenshot).
Every number also reproduces bit-for-bit on Python 3.13.5 with NumPy 2.3.5 and pandas 3.0.1.

```bash
python -m pip install -r requirements.txt
python monte_carlo.py      # the 50-line core; prints metrics, writes results/backtest.csv
python validate.py         # 11 checks; writes results/validation.json
python baselines.py        # audit-driven baselines, quarters, bootstrap; writes results/baselines.json
python error_allowance.py  # Part 2: regression errors and allowances; writes results/error_allowance_*
python book_exercise/hellwig_jarque_bera_book.py   # book example, fictitious data
```

Part 2 was run on Python 3.13.5, NumPy 2.2.6, pandas 2.3.1, SciPy 1.16.3, Matplotlib 3.10.3.

`monte_carlo.py` is exactly **50 physical lines** including imports, blank lines and I/O;
`validate.py` asserts that count. The checks: 731 unique consecutive dates, `cnt` equals
`casual + registered`, loss arithmetic on a hand case, two runs identical, test days in date
order from 2012-01-01 to 2012-12-31, no future leakage at two cutoffs, four days' exact
decisions re-enumerated in plain Python, a constant-demand edge case, invalid inputs rejected,
three extra seeds. Logs of every run are in `results/`.

![Run in VS Code](screenshots/vscode_run.png)

![Part 2 run in VS Code](screenshots/vscode_error_allowance.png)

![Error diagnostics](results/error_diagnostics.png)

## Practical limits and next step

Observed rentals are constrained by bike availability; they are not unmet demand. Ninety
equally weighted days ignore season, weather, growth and serial dependence, and the Q4 result
shows the cost of that. Resampling cannot produce a value absent from its window. In
production, start with the one-line quantile, shorten the window, add calendar and weather
forecasts and an explicit service target, calibrate real shortage and idle costs with
operations, and evaluate on a fresh period. Nothing here establishes a real-world saving.

## Files

`monte_carlo.py` (50-line core), `validate.py`, `baselines.py`, `error_allowance.py` (Part 2),
`book_exercise/` (book example on fictitious data), `data/day.csv`,
`data/provenance.json`, `results/` (backtest, metrics, validation, baselines, run logs),
`screenshots/vscode_run.png`, `screenshots/vscode_error_allowance.png`. Book pages and extracted book text are not redistributed.
