# Data

Raw data is **not** committed. Two sources:

## A. Live recorder (any coin, any date from now on) — `data/record.py`
The public WebSocket `trades` channel carries `users: [buyer, seller]` wallet addresses on every
print, so identity-tagged trade data can be collected for ANY listed coin without the archive.
```
pip install websockets pyarrow
python data/record.py --coins BTC,ETH,SOL,HYPE --hours 24          # trades + L2 book snapshots
python data/record.py --coins BTC --hours 1 --no-book --testnet    # quick check
```
Writes hourly parquet files to `data/raw/live/`. Run it on a machine that stays up (a cluster
login node or a cheap VM); a laptop that sleeps will produce gaps. Reconnects on drop.
What the public feed does NOT give: inventories, rejected orders, failed cancels, per-order
lifecycle. Those are only in the archive below.

## B. Archive (BTC/ETH/SOL Dec 2025 at order level; all coins Oct 2025–Jan 2026 at trade level)

## Source
Albers, Cucuringu, Howison, Shestopaloff (2026), *An Open Book: Level 4 Order Book
Data from the Hyperliquid Exchange*. Zenodo record 18184441
(https://doi.org/10.5281/zenodo.18184441). SSRN 6465720.

| Stream | Coverage | Size | Notes |
|---|---|---|---|
| Order statuses | BTC/ETH/SOL perps, Dec 2025 | ~880M records/day, 54-byte binary | includes rejected orders and failed cancels |
| Book diffs | BTC/ETH/SOL perps, Dec 2025 | ~50 GB tar (gzip JSONL) | every change to the visible book |
| Trades | all 250+ perps, Oct 2025 - Jan 2026 | | counterparties on both sides; spans 10 Oct 2025 cascade |

Files in the record (195 GB total): `trades_2025_{10,11,12}.tar`, `trades_2026_01.tar` (JSONL),
`{btc,eth,sol}_orders_202512.tar.xz` and `_rejected_` (54-byte binary records),
`book_diffs_202512.tar` (gzip JSONL), `mapdir.tar.xz` (lookup tables), `read_data.py`, `SCHEMA.md`.
**Start with `trades_2025_10.tar` (10 GB)**: it contains the 10 Oct cascade and is enough for
the wallet classifier, pair matrix, markouts and turn-taking statistics on all 250+ coins.
Download from the Zenodo record in a browser (the container cannot reach zenodo.org); then
read `SCHEMA.md` and map the trade fields onto the canonical frame below.

## Canonical trades frame (what every analysis module consumes)
```
time_ms, coin, side, px, sz, buyer, seller      # side is the TAKER's side
```
`analysis/wallets.py` converts this into maker/taker roles, per-wallet features, the MM flag,
the MM-MM trade-pair matrix (avoidance ratio), and the leader-switch (turn-taking) statistic.
Tested on a synthetic tape with known ground truth: `tests/test_wallets.py`.

## Layout
```
data/raw/live/       recorder output, hourly parquet (gitignored)
data/raw/zenodo/     archive downloads (gitignored)
data/processed/      canonical parquet, one file per (coin, day) (gitignored)
data/checksums.txt   sha256 of every raw file
```

## First week target
One week of BTC: identify MM wallets, estimate alpha from price impact,
arrival rates, spread distribution. See analysis/wallets.py, analysis/markouts.py.
