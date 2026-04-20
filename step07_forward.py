"""Step 7: forward returns, plus the backward return kept as a leakage contrast."""
import numpy as np
import pandas as pd

from features import (
    FEATURES_CACHE, HORIZONS, add_backward_returns, add_forward_returns,
)


def main():
    book = pd.read_pickle(FEATURES_CACHE)
    book = add_forward_returns(book)
    book = add_backward_returns(book)
    n = len(book)
    mid = book["mid"].to_numpy()

    print(f"\n{'=' * 78}\nSTEP 7 - FORWARD RETURNS\n{'=' * 78}")

    #the last N rows have no future left, they must stay NaN and be excluded
    for h in HORIZONS:
        na = book[f"fwd_return_{h}"].isna()
        expected = np.zeros(n, dtype=bool)
        expected[-h:] = True
        assert np.array_equal(na.to_numpy(), expected), (
            f"fwd_return_{h} NaN pattern wrong: {int(na.sum())} NaN, expected {h} at the tail"
        )
        print(f"[PASS] fwd_return_{h:<3} is NaN for exactly the last {h} rows and no others")

    #rebuild by hand off mid. sample only rows where the mid actually moved,
    #otherwise the check is vacuous: 0 == 0 passes even with the shift reversed.
    zero_frac = (book["fwd_return_10"] == 0).mean()
    print(f"\nrows where mid is unchanged over 10 events: {zero_frac:.2%}")
    print("sampling only rows that moved, so the check can actually fail:\n")
    rng = np.random.default_rng(7)
    movers = np.flatnonzero((book["fwd_return_10"] != 0).to_numpy()
                            & (np.arange(n) >= 10) & (np.arange(n) < n - 10))
    picks = np.sort(rng.choice(movers, size=5, replace=False))
    print(f"{'row':>9}{'mid[t-10]':>12}{'mid[t]':>10}{'mid[t+10]':>12}"
          f"{'by hand':>13}{'from column':>14}{'match':>7}")
    all_match = True
    for i in picks:
        by_hand = (mid[i + 10] - mid[i]) / mid[i]
        from_col = book["fwd_return_10"].iloc[i]
        ok = np.isclose(by_hand, from_col, rtol=0, atol=1e-15)
        all_match &= ok
        print(f"{i:>9,}{mid[i - 10]:>12.4f}{mid[i]:>10.4f}{mid[i + 10]:>12.4f}"
              f"{by_hand:>13.3e}{from_col:>14.3e}{str(ok):>7}")
    assert all_match, "hand reconstruction disagrees with the shifted column"
    print("[PASS] all five agree to floating point tolerance")

    #shift(-N) has to read ahead, not behind. the backward version on the same
    #rows must disagree, otherwise the direction is untested.
    backward = (mid[picks] - mid[picks - 10]) / mid[picks - 10]
    forward = book["fwd_return_10"].to_numpy()[picks]
    assert not np.allclose(forward, backward), (
        "forward matches backward on these rows, shift direction is unverified"
    )
    print("[PASS] forward differs from backward on the sampled rows, shift reads ahead")

    print("\nmagnitude check (should be a few basis points, not percent scale):")
    for h in HORIZONS:
        s = book[f"fwd_return_{h}"].std()
        print(f"  std(fwd_return_{h:<3}) = {s:.3e}  = {s * 1e4:7.3f} bps")
    s100 = book["fwd_return_100"].std() * 1e4
    assert 0.1 < s100 < 50, f"fwd_return_100 std is {s100:.2f} bps, horizon or alignment is wrong"
    print("[PASS] magnitudes are basis point scale")

    print(f"\n{'=' * 78}\nLEAKAGE CONTRAST - forward vs backward correlation with imb\n{'=' * 78}")
    print("bad_return is the BACKWARD return. It is a description of what already")
    print("happened, never a prediction. Reported as the contrast case.\n")
    print(f"{'horizon':<10}{'corr(imb, fwd)':>18}{'corr(imb, backward)':>22}")
    print("-" * 50)
    for h in HORIZONS:
        cf = book["imb"].corr(book[f"fwd_return_{h}"])
        cb = book["imb"].corr(book[f"bad_return_{h}"])
        print(f"N={h:<8}{cf:>18.5f}{cb:>22.5f}")

    book.to_pickle(FEATURES_CACHE)
    print(f"\nwrote {FEATURES_CACHE.name}")


if __name__ == "__main__":
    main()
