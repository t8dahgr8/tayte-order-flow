"""Config and file loading for the INTC order flow imbalance study."""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

TICKER = "INTC"
DATE = "2012-06-21"
LEVELS = 10

#lobster names files TICKER_DATE_START_END_type_LEVEL.csv, times in ms after midnight
_STEM = f"{TICKER}_{DATE}_34200000_57600000"
MESSAGE_FILE = DATA_DIR / f"{_STEM}_message_{LEVELS}.csv"
ORDERBOOK_FILE = DATA_DIR / f"{_STEM}_orderbook_{LEVELS}.csv"

MESSAGE_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]

MESSAGE_DTYPES = {
    "time": "float64",
    "event_type": "int64",
    "order_id": "int64",
    "size": "int64",
    "price": "int64",
    "direction": "int64",
}


def orderbook_cols(levels=LEVELS):
    """40 column names from the repeating 4-per-level pattern."""
    cols = []
    for level in range(1, levels + 1):
        cols += [
            f"ask_price_{level}",
            f"ask_size_{level}",
            f"bid_price_{level}",
            f"bid_size_{level}",
        ]
    return cols


ORDERBOOK_COLS = orderbook_cols()

ASK_PRICE_COLS = [f"ask_price_{L}" for L in range(1, LEVELS + 1)]
ASK_SIZE_COLS = [f"ask_size_{L}" for L in range(1, LEVELS + 1)]
BID_PRICE_COLS = [f"bid_price_{L}" for L in range(1, LEVELS + 1)]
BID_SIZE_COLS = [f"bid_size_{L}" for L in range(1, LEVELS + 1)]

#padding written into unoccupied level slots
ASK_SENTINEL = 9999999999
BID_SENTINEL = -9999999999


def load_raw():
    """Read both csvs. Explicit dtypes so anything non-numeric blows up here, not later."""
    messages = pd.read_csv(
        MESSAGE_FILE, header=None, names=MESSAGE_COLS, dtype=MESSAGE_DTYPES
    )
    orderbook = pd.read_csv(
        ORDERBOOK_FILE, header=None, names=ORDERBOOK_COLS, dtype="int64"
    )
    return messages, orderbook
