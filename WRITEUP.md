# Order Flow Imbalance at the Touch

**Does size imbalance at the best bid/ask predict the short-horizon change in the mid-price?**

INTC, 2012-06-21, LOBSTER level-10 message and orderbook files. One ticker, one day.
This is a measurement study, not a strategy and not a backtest.

---

## 1. Question

At any instant a limit order book holds a queue of buy orders at the best bid and a
queue of sell orders at the best ask. When those queues are lopsided — far more size
resting on one side than the other — does the mid-price tend to drift toward the heavier
side over the next few events? I measure the strength of that relationship, correct the
standard errors for the overlapping-window problem it creates, and then ask the only
question that matters economically: is the predicted move larger than the cost of
crossing the spread to capture it?

## 2. Data

| | |
|---|---|
| Ticker / date | INTC, 2012-06-21 (NASDAQ ITCH via LOBSTER) |
| Levels | 10 |
| Events | 624,040 |
| Session | 09:30:00.0057 – 15:59:59.9484 |
| Mid range | $26.615 – $27.595 (open 27.515 to close 26.715, −2.91%) |

Event composition:

| Type | Meaning | Count | Share |
|---|---|---|---|
| 1 | New visible limit order | 304,790 | 48.84% |
| 2 | Partial cancellation | 8,028 | 1.29% |
| 3 | Total deletion | 278,739 | 44.67% |
| 4 | Visible execution | 28,924 | 4.63% |
| 5 | Hidden execution | 3,559 | 0.57% |

**94.8% of the day is order placement and retraction; only 5.2% is stock changing
hands.** There are roughly 18 order-management events per trade, and total deletions run
at 91% of new submissions — most orders posted to this book were cancelled, never
executed.

**Filtering decisions.** Rows outside 09:30–16:00 and events of type 6 (cross/auction) or
7 (halt) were to be dropped. *Zero rows met either condition*: LOBSTER had already cut
the file to the continuous session (the filename encodes 34200000–57600000 ms), and INTC
neither halted nor crossed that day. The filters are retained as correct-but-inert and
reported as such rather than quietly omitted. Likewise, no level slot in 12,480,800
carried LOBSTER's ±9999999999 padding — INTC's book held at least 10 occupied levels on
both sides for the entire day — so the sentinel handling never fired on real data and is
instead verified against a synthetic padded book in `src/test_sentinels.py`. No crossed
books, no locked books, no empty sides. Nothing was dropped at any stage.

**Tick regime.** The spread was exactly one cent for 76.70% of events and two cents for
23.23%, median one cent, while the median queue held 13,391 shares at the bid and 12,291
at the ask. These two facts are the same fact. At $27 a one-cent tick is a coarse
increment, so a trader who wants to bid more aggressively cannot improve by half a cent —
the only available move is to join the back of the queue. Competition that would
otherwise narrow the spread accumulates as depth instead. This is the large-tick regime,
and it is why queue imbalance is informative here: when the price is constrained from
moving, information accumulates in the queue.

## 3. Method

**Alignment.** Row *k* of the message file describes the event that moved the book from
orderbook row *k−1* to row *k*; orderbook row *k* is therefore the state **immediately
after** message *k*. This was not assumed. For 2,000 sampled add/cancel/delete/execute
events I predicted the visible size at the event's own price level from the previous
orderbook row and compared it to row *k*: **100.00% agreement**. Under the off-by-one
alternative the same test scores **4.91%**. Both `imb[t]` and `mid[t]` are read from this
same post-event state, so both are known at *t*.

**Imbalance.** `imb = (bid_size − ask_size) / (bid_size + ask_size)`, bounded in [−1, +1],
computed at level 1 only. Mean +0.0601, standard deviation 0.4158, and 6.10% of rows
exceed |imb| > 0.8.

**Forward return.** For horizon *N* events,
`fwd_return_N[t] = (mid[t+N] − mid[t]) / mid[t]`, for N in {10, 50, 100}. The window is
strictly forward, produced by a forward shift; the final *N* rows are undefined, set NaN,
and excluded rather than filled. Verification: the NaN pattern covers exactly the last
*N* rows and no others; five randomly chosen rows were reconstructed by direct
hand-indexing of `mid` and matched to floating-point tolerance. Those five were drawn
**only from rows where the mid actually moved** — 95.45% of rows show no change over 10
events, so a sample drawn at random would have compared 0 against 0 and passed even with
the shift reversed. The forward and backward values were additionally asserted to differ
on those same rows, which is what pins the direction. Return magnitudes are basis-point
scale (std 0.430 / 0.968 / 1.397 bps for N = 10 / 50 / 100), not percent scale.

**Why HAC standard errors are required.** Consecutive observations share almost their
entire forward window: rows *t* and *t+1* at N = 100 have 99 of 100 events in common. The
residuals are therefore severely autocorrelated, violating the independence assumption
behind the naive OLS standard error. The naive error is far too small and the naive
t-statistic is meaningless. Newey-West (HAC) errors, with lag length set to *N*, account
for that overlap. The size of the gap is itself a result and is reported below.

## 4. Results

**Deciles.** Mean forward return rises monotonically across all ten imbalance deciles at
every horizon, with no reversals (`fig08_deciles.png`). At N = 100 the bottom
decile averages −1.259 bps and the top decile +1.002 bps, on ~62,400 observations each,
with ±1.96 SE bands far narrower than the spacing between deciles.

**Regression** — `fwd_return_N ~ imb`:

| Horizon | n | coef (bps) | naive SE (bps) | **naive t** | HAC SE (bps) | **HAC t** | R² | inflation |
|---|---|---|---|---|---|---|---|---|
| N = 10 | 624,030 | 0.252 | 0.0013 | 197.9 | 0.0052 | 48.6 | 0.0591 | 4.1x |
| N = 50 | 623,990 | 0.901 | 0.0027 | 331.5 | 0.0190 | 47.3 | 0.1497 | 7.0x |
| N = 100 | 623,940 | 1.461 | 0.0038 | 381.2 | 0.0336 | 43.5 | 0.1889 | 8.8x |

The naive t-statistics overstate significance by 4.1x to 8.8x, and the distortion grows
with horizon exactly as the overlap mechanism predicts. The effect remains overwhelmingly
significant after correction; the correction changes the honest magnitude of the claim,
not its sign.

**On the R².** The specification anticipated an R² of a fraction of one percent, with a
large R² flagged as evidence of leakage. The observed R² is 18.9% at N = 100. I attribute
this to the tick regime rather than to a defect, on four grounds: the alignment test
discriminates at 100% versus 4.91%; the forward shift was hand-verified on rows with
genuine price movement and asserted to differ from its backward counterpart; the placebo
below collapses to nothing; and in a large-tick name the mid is confined to a half-cent
grid and is unchanged for 95.45% of 10-event windows, so the dependent variable is close
to a three-state outcome that queue depletion mechanically anticipates. The same study on
a small-tick stock would be expected to produce the far smaller R² the specification
describes. This is a consequence of the ticker choice and is stated as such, not as a
general finding.

**Tradability.** Median spread 1.000 cents, so crossing to enter costs 0.500 cents.

| | |
|---|---|
| Mean `fwd_return_100`, top imbalance decile | +1.0019 bps |
| Mean price in that decile | $27.1208 |
| **Predicted edge** | **+0.2717 cents** |
| **Cost of crossing (half-spread)** | **0.5000 cents** |
| Ratio | 0.543 |
| Shortfall | 0.2283 cents |

**The predicted move is roughly half the cost of crossing the spread to capture it.** The
signal is real and statistically overwhelming and is not tradable by taking liquidity. It
is exploitable only by a participant already resting in the queue, who pays no spread to
enter and whose decision is whether to hold or abandon queue position — which is
precisely why this effect belongs to passive market makers rather than to aggressive
traders.

**Robustness.** All four checks survive.

| Check | Result at N = 100 |
|---|---|
| Open 09:30–10:30 | coef 1.621, HAC t 19.7, R² 0.142 |
| Midday 10:30–15:00 | coef 1.434, HAC t 35.2, R² 0.223 |
| Close 15:00–16:00 | coef 1.388, HAC t 21.7, R² 0.254 |
| One-cent spreads only (76.7% of rows) | coef 1.434, HAC t 41.3, R² 0.207 |
| Excluding ±10 events around any execution | coef 1.039, HAC t 32.0, R² 0.113 |
| **Shuffled `imb` (placebo)** | **coef 0.002, HAC t 0.5, R² 0.0000003** |

The effect is *not* concentrated at the open — the coefficient is largest there but the
fit is tightest at midday and into the close. Restricting to one-cent spreads leaves the
coefficient essentially unchanged (1.434 vs 1.461), so the result is not a mechanical
artifact of wider spreads. Removing every row within 10 events of an execution weakens
the effect materially but does not remove it: at N = 100 the coefficient falls 29% and R²
falls from 0.189 to 0.113, while at N = 10 it nearly vanishes (coefficient 0.252 to
0.051, R² 0.059 to 0.005). The very-short-horizon effect is therefore substantially
trade-driven, while the longer-horizon effect persists in quiet stretches of the book.
The placebo collapses R² by a factor of roughly 566,000.

## 5. Threats to validity

- **One ticker, one day.** Nothing here supports a claim of generality. The tick regime
  alone plausibly drives the effect size, and INTC fell 2.91% on this date, so a single
  directional day is embedded in the sample.
- **Overlapping observations.** Addressed with HAC errors, but HAC is an asymptotic
  correction, not a cure; the effective sample size is far below 624,040.
- **Tick-size mechanics as the driver.** In a large-tick book the mid is pinned to a
  half-cent grid and moves only when a touch queue is exhausted. Imbalance may be
  measuring proximity to queue exhaustion rather than anything about information or
  intent. The one-cent-spread check does not separate these, because that subsample *is*
  the tick-constrained regime.
- **No queue-position modeling.** The tradability comparison assumes a market maker
  enters at no spread cost. Real passive execution depends on queue priority, fill
  probability, and adverse selection on fills, none of which are modeled. No claim about
  executable P&L is made or supported.
- **Event time, not clock time.** Horizons are counted in events, so N = 100 spans about
  3.7 seconds on average but varies with activity, and is systematically shorter near the
  open.
- **Level 1 only.** Depth beyond the touch is discarded, though it is present in the data.

## 6. The leakage contrast

The specification anticipated that imbalance would correlate far more strongly with the
*backward* return `(mid[t] − mid[t−N]) / mid[t−N]` than with the forward return, since
imbalance at *t* partly reflects what just happened. **That is not what this data shows:**

| Horizon | corr(imb, forward) | corr(imb, backward) |
|---|---|---|
| N = 10 | +0.2430 | **−0.1048** |
| N = 50 | +0.3869 | +0.0146 |
| N = 100 | +0.4346 | +0.0854 |

The backward correlation is weaker at every horizon and *negative* at N = 10. The
mechanism is specific to a large-tick book: when the ask queue is exhausted and the mid
ticks up, the new best ask is a fresh, deep level while the new best bid is the thin,
newly-formed queue left behind. Imbalance therefore flips *against* the move that just
occurred. Imbalance in this regime is genuinely anticipatory rather than descriptive.

The exercise still does its job. The backward return remains a quantity that is trivially
easy to compute, looks like a predictor, and is not one — it is a description of the past.
The distinction that matters is not which correlation is larger but which observations
were available at time *t*: `imb[t]` and `mid[t]` are known then, `mid[t+N]` is not, and
`mid[t−N]` cannot be used to predict something that has already happened.

---

## Conclusion

Queue imbalance at the touch is a real, monotone, and statistically overwhelming predictor
of short-horizon mid-price movement in INTC on this date, surviving time-of-day splits,
spread conditioning, post-trade exclusion, and a shuffle placebo, with HAC t-statistics
between 43 and 49 across horizons. Its economic magnitude — 0.27 cents against a
0.50-cent cost of crossing — places it out of reach of anyone who must pay the spread to
act on it.
