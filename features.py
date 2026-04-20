"""Imbalance, forward returns, and the deliberately wrong backward returns."""
from lobster import OUTPUT_DIR

HORIZONS = (10, 50, 100)
FEATURES_CACHE = OUTPUT_DIR / "features.pkl"


def add_imbalance(book):
    """imb in [-1,1]. +1 means all the size is on the bid, -1 all on the ask."""
    book = book.copy()
    book["imb"] = ((book["bid_size"] - book["ask_size"])
                   / (book["bid_size"] + book["ask_size"]))
    book["spread_cents"] = (book["spread"] * 100).round().astype(int)
    return book


def add_forward_returns(book, horizons=HORIZONS):
    """mid N events ahead. Strictly forward, last N rows go NaN and stay NaN."""
    book = book.copy()
    mid = book["mid"]
    for n in horizons:
        fwd = mid.shift(-n)
        book[f"fwd_mid_{n}"] = fwd
        book[f"fwd_return_{n}"] = (fwd - mid) / mid
    return book


def add_backward_returns(book, horizons=HORIZONS):
    """Backward looking on purpose. This is the leakage contrast, never a predictor."""
    book = book.copy()
    mid = book["mid"]
    for n in horizons:
        past = mid.shift(n)
        book[f"bad_return_{n}"] = (mid - past) / past
    return book
