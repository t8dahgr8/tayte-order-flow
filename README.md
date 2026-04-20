# Order Flow Imbalance at the Touch

Does size imbalance at the best bid/ask predict the short-horizon change in the
mid-price? A measurement study on one ticker, one trading day, using LOBSTER level-10
limit order book data.

**INTC, 2012-06-21 — 624,040 events.**

## Result

Queue imbalance at the touch predicts short-horizon mid-price movement monotonically
across all ten deciles and all three horizons, with HAC t-statistics between 43 and 49.
It is also **not tradable by crossing the spread**:

| | |
|---|---|
| Predicted move, top imbalance decile, 100 events ahead | **0.272 cents** |
| Cost of crossing the spread to enter | **0.500 cents** |

The signal is real and roughly half the size of the toll you would pay to act on it. It
belongs to participants already resting in the queue, who pay no spread to enter and
whose decision is whether to hold or abandon queue position.

Full analysis and caveats in [WRITEUP.md](WRITEUP.md).

## Running it

```bash
pip install -r requirements.txt
python download_data.py
python run_all.py
```

The LOBSTER CSVs total ~184 MB and the orderbook file alone exceeds GitHub's 100 MB
limit, so `data/` is gitignored and `download_data.py` pulls the files on demand.

Steps can also be run individually, in order. Steps 2–4 cache a working frame to
`output/top_of_book.pkl` so later steps do not re-parse the 158 MB orderbook file.

## Layout

```
./
  lobster.py            paths, column names, raw loader
  pipeline.py           steps 2-4 transformations
  features.py           imbalance and forward/backward returns
  regress.py            OLS with naive and Newey-West errors

  step01_load.py        load + prove the two files are row aligned
  steps_02_04.py        units, session filter, top of book
  step05_plots.py       mid, spread, spread histogram
  step06_imbalance.py   compute imb, distribution
  step07_forward.py     forward returns + the leakage contrast
  step08_deciles.py     mean forward return by imbalance decile
  step09_regression.py  naive vs HAC standard errors
  step10_tradable.py    predicted edge vs half-spread
  step11_robustness.py  time of day, spread, post-trade, shuffle placebo

  test_sentinels.py     synthetic check of the padded-level handling
  download_data.py      fetch the CSVs
  run_all.py            run every step in order
```

## Notes on correctness

Three steps in this pipeline can fail silently — producing results that look better
rather than throwing an error. Each is verified rather than assumed:

**Row alignment.** Orderbook row *k* is the state *after* message *k*. Verified by
predicting the visible size at each event's own price level from the previous orderbook
row: 100.00% agreement on 2,000 sampled events, versus 4.91% under the off-by-one
alternative.

**Forward-shift direction.** The five hand-reconstruction checks sample only rows where
the mid actually moved. 95.45% of rows show no change over 10 events, so a random sample
would compare 0 against 0 and pass even with the shift reversed. Forward and backward
values are additionally asserted to differ on those rows.

**Overlapping windows.** Consecutive observations share nearly their whole forward
window, so naive OLS standard errors are far too small. Newey-West errors with lag
length *N* shrink the t-statistics by 4.1x to 8.8x. Both are reported.

**Placebo.** Shuffling `imb` collapses R² from 0.189 to 0.0000003 and the HAC t-statistic
from 43.5 to 0.5.

Two of the specification's stated expectations did not hold on this ticker — the R² is
18.9% rather than a fraction of a percent, and imbalance correlates *more* strongly with
forward returns than backward ones. Both are consequences of INTC being a large-tick
stock, and both are discussed rather than smoothed over in the write-up.

## Data

LOBSTER sample files, mirrored at
[totalorganfailure/lobster-data](https://huggingface.co/datasets/totalorganfailure/lobster-data).
Original source: [LOBSTER](https://lobsterdata.com) — Limit Order Book System, Humboldt
University Berlin.


Author: Tayte Choudhury.

Revalidated locally on November 12, 2026. Actual available INTC results differ from the resume figures; this repository preserves the measured dataset and results. Raw LOBSTER CSVs are excluded. The current pipeline filters event rows marked as halt/cross, but does not propagate halt state across subsequent rows; use additional stateful halt handling before applying it to sessions with prolonged halts.


Portfolio: https://t8c4.com/projects/order-flow.html

Repository: https://github.com/t8dahgr8/tayte-order-flow


The complete original folder layout, public presentation and Render configuration are included in project-bundle.zip. Extract that archive before following instructions referring to public/.
