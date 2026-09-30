"""Download the Hyperliquid L4 dataset from Zenodo into data/raw/ and verify checksums.

Usage: python data/download.py [--only trades|orders|book] [--coin BTC]

The Zenodo record is large (tens of GB).  Start with the trades stream.
"""
import argparse, hashlib, sys
from pathlib import Path

ZENODO_RECORD = "18184441"
RAW = Path(__file__).parent / "raw"

def sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["trades", "orders", "book"], default="trades")
    ap.add_argument("--coin", default="BTC")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Fetch file list from https://zenodo.org/api/records/{ZENODO_RECORD} and download the "
          f"'{args.only}' stream for {args.coin} into {RAW}.")
    print("TODO: implement with `requests` once the exact file names are confirmed; "
          "then append sha256 lines to data/checksums.txt.")
    sys.exit(0)

if __name__ == "__main__":
    main()
