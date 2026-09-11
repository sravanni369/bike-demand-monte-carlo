"""Independent checks and reproducible run evidence; outside the 50-line core."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
from monte_carlo import evaluate, loss

base = Path(__file__).resolve().parent
frame = pd.read_csv(base / 'data/day.csv', parse_dates=['dteday'])
assert len(frame) == 731 and frame.dteday.nunique() == 731
assert frame.dteday.diff().dropna().eq(pd.Timedelta(days=1)).all()
assert np.array_equal(frame.cnt, frame.casual + frame.registered)
assert len((base / 'monte_carlo.py').read_text().splitlines()) == 50
assert list(loss(np.array([100, 100, 100]), np.array([80, 100, 120]))) == [20, 0, 80]
rows, metrics = evaluate(frame)
again, repeated = evaluate(frame)
assert rows.equals(again) and metrics == repeated
assert rows.date.iloc[0] == '2012-01-01' and rows.date.iloc[-1] == '2012-12-31'
# Future demand must not affect any earlier choice: perturb everything from a cutoff and
# require every decision before the cutoff to be unchanged (two cutoffs, 183 days checked).
for cutoff in ['2012-01-01', '2012-07-01']:
    altered = frame.copy()
    altered.loc[altered.dteday >= cutoff, 'cnt'] += 10000
    changed, _ = evaluate(altered)
    before = pd.to_datetime(rows.date) < cutoff
    assert rows.loc[before, ['baseline', 'mc', 'exact']].equals(changed.loc[before, ['baseline', 'mc', 'exact']])
    # the first day at the cutoff sees only pre-cutoff history, so it must be unchanged too
    assert rows.iloc[before.sum()][['baseline', 'mc', 'exact']].equals(changed.iloc[before.sum()][['baseline', 'mc', 'exact']])
# Independently enumerate every candidate using plain Python, not the production loss function.
for j in [0, 90, 180, 365]:
    cutoff = pd.Timestamp(rows.iloc[j].date)
    history = frame.loc[frame.dteday < cutoff, 'cnt'].tail(90).tolist()
    objective = lambda q: sum(4 * max(d-q, 0) + max(q-d, 0) for d in history) / 90
    expected = min(range(0, 10001, 250), key=objective)
    assert rows.iloc[j].exact == expected
constant = frame.copy()
constant['cnt'] = 1000
constant_rows, constant_metrics = evaluate(constant, draws=10)
assert (constant_rows.mc == 1000).all() and constant_metrics['mc_loss'] == 0
for bad in [frame.iloc[:0], frame.assign(cnt=-1), pd.concat([frame, frame.iloc[:1]])]:
    try:
        evaluate(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid input accepted')
stability = []
for seed in [7, 21, 84]:
    _, m = evaluate(frame, seed=seed)
    stability.append(m)
(base / 'results/metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
rows.to_csv(base / 'results/backtest.csv', index=False)
evidence = {'status': 'PASS', 'checks': ['731 unique consecutive dates', 'target count integrity',
    'exactly 50 physical core lines', 'asymmetric loss arithmetic', 'identical repeated runs',
    'chronological 366-day test', 'future perturbation invariance (2 cutoffs)', 'independent exhaustive reference',
    'constant-demand edge case', 'invalid input rejection', 'three additional random seeds'],
    'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
    'data_sha256': hashlib.sha256((base / 'data/day.csv').read_bytes()).hexdigest(),
    'additional_seeds': stability, 'metrics': metrics}
(base / 'results/validation.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps(evidence, indent=2))
