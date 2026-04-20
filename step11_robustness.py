"""Step 11: does the result survive being attacked?"""
import numpy as np
import pandas as pd

from features import FEATURES_CACHE, HORIZONS
from regress import fit_ols, print_table


def main():
    book = pd.read_pickle(FEATURES_CACHE)
    print(f"\n{'=' * 78}\nSTEP 11 - ROBUSTNESS\n{'=' * 78}")

    #1 time of day
    print(f"\n{'-' * 78}\n1. TIME OF DAY SPLIT\n{'-' * 78}")
    windows = [("open 0930-1030", 34200, 37800),
               ("midday 1030-1500", 37800, 54000),
               ("close 1500-1600", 54000, 57600)]
    for label, lo, hi in windows:
        part = book[(book["time"] >= lo) & (book["time"] < hi)]
        rows = [(f"N={h}", fit_ols(part, f"fwd_return_{h}", "imb", maxlags=h))
                for h in HORIZONS]
        print_table(rows, f"{label}   ({len(part):,} rows)")

    #2 spread conditioning, wide spreads could drive it mechanically
    print(f"\n{'-' * 78}\n2. ONE CENT SPREAD ONLY\n{'-' * 78}")
    narrow = book[book["spread_cents"] == 1]
    rows = [(f"N={h}", fit_ols(narrow, f"fwd_return_{h}", "imb", maxlags=h))
            for h in HORIZONS]
    print_table(rows, f"spread == 1 cent   ({len(narrow):,} rows, "
                      f"{len(narrow) / len(book):.1%} of sample)")

    #3 away from trades
    print(f"\n{'-' * 78}\n3. POST-TRADE EXCLUSION\n{'-' * 78}")
    is_exec = (book["event_type"] == 4).astype(float)
    near_trade = is_exec.rolling(21, center=True, min_periods=1).max() > 0
    away = book[~near_trade]
    rows = [(f"N={h}", fit_ols(away, f"fwd_return_{h}", "imb", maxlags=h))
            for h in HORIZONS]
    print_table(rows, f"dropping rows within 10 events of an execution   "
                      f"({len(away):,} rows kept, {int(near_trade.sum()):,} dropped)")

    #4 the control that has to fail
    print(f"\n{'-' * 78}\n4. SHUFFLE CONTROL\n{'-' * 78}")
    print("imb is randomly permuted, destroying any real link to the future.")
    print("The coefficient must collapse to zero. If it does not, there is a bug.\n")
    shuffled = book.copy()
    rng = np.random.default_rng(11)
    shuffled["imb"] = rng.permutation(shuffled["imb"].to_numpy())
    rows = [(f"N={h}", fit_ols(shuffled, f"fwd_return_{h}", "imb", maxlags=h))
            for h in HORIZONS]
    print_table(rows, "shuffled imb (placebo)")

    real = fit_ols(book, "fwd_return_100", "imb", maxlags=100)
    fake = fit_ols(shuffled, "fwd_return_100", "imb", maxlags=100)
    print(f"\nN=100  real R2 = {real['r2']:.5f}   shuffled R2 = {fake['r2']:.8f}"
          f"   ratio = {real['r2'] / max(fake['r2'], 1e-12):,.0f}x")
    print(f"N=100  real HAC t = {real['t_hac']:.1f}   shuffled HAC t = {fake['t_hac']:.2f}")


if __name__ == "__main__":
    main()
