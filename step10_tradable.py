"""Step 10: is the predicted move bigger than the cost of crossing?"""
import pandas as pd

from features import FEATURES_CACHE


def main():
    book = pd.read_pickle(FEATURES_CACHE)
    top = book[book["imb_decile"] == 10]
    bottom = book[book["imb_decile"] == 1]

    median_spread_cents = book["spread"].median() * 100
    half_spread_cents = median_spread_cents / 2

    print(f"\n{'=' * 78}\nSTEP 10 - IS IT TRADABLE?\n{'=' * 78}")
    print(f"median spread          : {median_spread_cents:.3f} cents")
    print(f"half spread (cost in)  : {half_spread_cents:.3f} cents")
    print("\npredicted move over the next 100 events, by extreme decile:\n")
    print(f"{'decile':<10}{'rows':>10}{'mean ret (bps)':>18}{'mean price':>13}{'edge (cents)':>15}")
    print("-" * 66)

    results = {}
    for name, part, sign in [("10 (top)", top, +1), ("1 (bottom)", bottom, -1)]:
        r = part["fwd_return_100"].mean()
        price = part["mid"].mean()
        edge = r * price * 100
        results[name] = edge
        print(f"{name:<10}{len(part):>10,}{r * 1e4:>18.4f}{price:>13.4f}{edge:>15.4f}")

    edge_top = results["10 (top)"]
    print(f"\n{'-' * 66}")
    print(f"predicted edge, top decile : {edge_top:+.4f} cents")
    print(f"cost of crossing to enter  : {half_spread_cents:.4f} cents")
    print(f"ratio edge / half spread   : {edge_top / half_spread_cents:.3f}")
    print(f"shortfall                  : {half_spread_cents - edge_top:.4f} cents")

    if edge_top < half_spread_cents:
        print("\nCONCLUSION: the predicted move is SMALLER than the cost of crossing.")
        print("The signal is real but not tradable by taking liquidity. Only a")
        print("participant already resting in the queue, who pays no spread to enter,")
        print("can act on it, by managing queue position rather than by crossing.")
    else:
        print("\nCONCLUSION: edge exceeds the half spread. Check for leakage before believing it.")


if __name__ == "__main__":
    main()
