"""Fetch the INTC sample files from the hugging face mirror.

The csvs are too big to commit (158MB orderbook), so they are gitignored and
pulled on demand instead.
"""
import urllib.request

from lobster import DATA_DIR, MESSAGE_FILE, ORDERBOOK_FILE

REPO = "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main"
FOLDER = "LOBSTER_SampleFile_INTC_2012-06-21_10"


def fetch(target):
    if target.exists():
        print(f"already have {target.name} ({target.stat().st_size:,} bytes)")
        return
    url = f"{REPO}/{FOLDER}/{target.name}"
    print(f"downloading {target.name} ...")
    #hf drops requests with no user agent
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(target, "wb") as f:
        f.write(r.read())
    print(f"  saved {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    fetch(MESSAGE_FILE)
    fetch(ORDERBOOK_FILE)
