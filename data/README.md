# Data

Raw data is **not** committed. Populate `data/raw/` with `python data/download.py`.

## Source
Albers, Cucuringu, Howison, Shestopaloff (2026), *An Open Book: Level 4 Order Book
Data from the Hyperliquid Exchange*. Zenodo record 18184441
(https://doi.org/10.5281/zenodo.18184441). SSRN 6465720.

| Stream | Coverage | Size | Notes |
|---|---|---|---|
| Order statuses | BTC/ETH/SOL perps, Dec 2025 | ~880M records/day, 54-byte binary | includes rejected orders and failed cancels |
| Book diffs | BTC/ETH/SOL perps, Dec 2025 | ~50 GB tar (gzip JSONL) | every change to the visible book |
| Trades | all 250+ perps, Oct 2025 - Jan 2026 | | counterparties on both sides; spans 10 Oct 2025 cascade |

## Layout
```
data/raw/        untouched downloads (gitignored)
data/processed/  parquet, one file per (coin, day) (gitignored)
data/checksums.txt   sha256 of every raw file
```

## First week target
One week of BTC: identify MM wallets, estimate alpha from price impact,
arrival rates, spread distribution. See analysis/wallets.py, analysis/markouts.py.
