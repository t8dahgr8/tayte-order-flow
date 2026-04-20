"""Step 9: OLS of forward return on imbalance, naive vs Newey-West errors."""
import pandas as pd

from features import FEATURES_CACHE, HORIZONS
from regress import fit_ols, print_table


def main():
    book = pd.read_pickle(FEATURES_CACHE)

    print(f"\n{'=' * 78}\nSTEP 9 - REGRESSION  fwd_return_N ~ imb\n{'=' * 78}")
    rows = [(f"N={h}", fit_ols(book, f"fwd_return_{h}", "imb", maxlags=h))
            for h in HORIZONS]
    print_table(rows, "naive OLS errors vs HAC (Newey-West, lags = N)")

    print("\nt-stat inflation from ignoring overlap:")
    for label, r in rows:
        print(f"  {label:<8} naive t = {r['t_naive']:9.1f}   HAC t = {r['t_hac']:8.1f}"
              f"   overstated by {r['t_naive'] / r['t_hac']:.1f}x")


if __name__ == "__main__":
    main()
