"""Record Hyperliquid public trades and L2 book snapshots to parquet, with wallet identities.

The public WebSocket `trades` channel carries `users: [buyer, seller]` for every print, so
identity-tagged trade data for ANY listed coin can be collected without the Zenodo tarball.
This recorder is the path to (a) coins beyond BTC/ETH/SOL for the cross-coin rent-vs-toxicity
test and (b) live data after January 2026.

What it writes (one parquet per hour per stream, under data/raw/live/):
    trades_<COIN>_<YYYYMMDDHH>.parquet
        time_ms, coin, side, px, sz, tid, hash, buyer, seller
    l2_<COIN>_<YYYYMMDDHH>.parquet       (one row per snapshot per level)
        time_ms, coin, side, level, px, sz, n_orders
    (the book feed is a snapshot per block, so this is Level 2 at block resolution)

Usage:
    python data/record.py --coins BTC,ETH,SOL,HYPE --hours 24
    python data/record.py --coins BTC --hours 1 --testnet     # for a quick check

Requires: pip install websockets pyarrow      (not in the base install on purpose)

Reconnects on drop. Rate limits: one connection, <= 1000 subscriptions, far under the
2000 msg/min send cap (we only send subscribe messages).  Reads carry no address-based cost.

Caveat: the public feed gives no inventory or rejected orders; those exist only in the
Zenodo L4 archive (BTC/ETH/SOL, Dec 2025).  Trade-level identity is enough for the wallet
classifier (analysis/wallets.py), the trade-pair matrix, and markouts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw" / "live"
MAINNET = "wss://api.hyperliquid.xyz/ws"
TESTNET = "wss://api.hyperliquid-testnet.xyz/ws"


class HourlyWriter:
    """Buffers rows per (stream, coin) and flushes to one parquet per hour."""

    def __init__(self, root: Path):
        import pyarrow as pa  # noqa: F401  (fail early if missing)
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.buf: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    @staticmethod
    def hour_key(time_ms: int) -> str:
        return datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).strftime("%Y%m%d%H")

    def add(self, stream: str, coin: str, row: dict):
        self.buf[(stream, coin, self.hour_key(row["time_ms"]))].append(row)

    def flush(self, only_closed_hours: bool = True):
        import pyarrow as pa
        import pyarrow.parquet as pq
        now_key = self.hour_key(int(time.time() * 1000))
        for key in list(self.buf):
            stream, coin, hk = key
            if only_closed_hours and hk >= now_key:
                continue
            rows = self.buf.pop(key)
            if not rows:
                continue
            path = self.root / f"{stream}_{coin}_{hk}.parquet"
            table = pa.Table.from_pylist(rows)
            if path.exists():                                   # append on reconnect within the hour
                table = pa.concat_tables([pq.read_table(path), table])
            pq.write_table(table, path, compression="zstd")
            print(f"wrote {path.name} ({table.num_rows} rows)", flush=True)


async def record(url: str, coins: list[str], hours: float, writer: HourlyWriter, book: bool):
    import websockets
    deadline = time.time() + hours * 3600
    n_trades = 0
    while time.time() < deadline:
        try:
            async with websockets.connect(url, ping_interval=20, max_size=2**24) as ws:
                for c in coins:
                    await ws.send(json.dumps({"method": "subscribe",
                                              "subscription": {"type": "trades", "coin": c}}))
                    if book:
                        await ws.send(json.dumps({"method": "subscribe",
                                                  "subscription": {"type": "l2Book", "coin": c}}))
                print(f"subscribed to {coins} on {url}", flush=True)
                last_flush = time.time()
                while time.time() < deadline:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                    ch = msg.get("channel")
                    if ch == "trades":
                        for t in msg["data"]:
                            users = t.get("users") or [None, None]
                            writer.add("trades", t["coin"], {
                                "time_ms": int(t["time"]), "coin": t["coin"], "side": t["side"],
                                "px": float(t["px"]), "sz": float(t["sz"]), "tid": int(t["tid"]),
                                "hash": t.get("hash"), "buyer": users[0], "seller": users[1]})
                            n_trades += 1
                    elif ch == "l2Book":
                        d = msg["data"]
                        tms = int(d["time"])
                        for side, levels in zip(("bid", "ask"), d["levels"]):
                            for lvl, L in enumerate(levels):
                                writer.add("l2", d["coin"], {
                                    "time_ms": tms, "coin": d["coin"], "side": side, "level": lvl,
                                    "px": float(L["px"]), "sz": float(L["sz"]), "n_orders": int(L["n"])})
                    if time.time() - last_flush > 300:
                        writer.flush()
                        last_flush = time.time()
                        print(f"... {n_trades} trades so far", flush=True)
        except (asyncio.TimeoutError, ConnectionError, OSError) as e:
            print(f"reconnect after {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(2)
        except Exception as e:                                   # websockets-specific closures
            if "ConnectionClosed" in type(e).__name__:
                print("reconnect after connection closed", flush=True)
                await asyncio.sleep(2)
            else:
                raise
    writer.flush(only_closed_hours=False)
    print(f"done: {n_trades} trades", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default="BTC,ETH,SOL")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--no-book", action="store_true", help="trades only (much smaller)")
    ap.add_argument("--testnet", action="store_true")
    ap.add_argument("--out", default=str(RAW))
    args = ap.parse_args()
    writer = HourlyWriter(Path(args.out))
    coins = [c.strip() for c in args.coins.split(",")]
    url = TESTNET if args.testnet else MAINNET
    loop = asyncio.new_event_loop()
    stop = lambda *_: (writer.flush(only_closed_hours=False), loop.stop())
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, stop)
    loop.run_until_complete(record(url, coins, args.hours, writer, book=not args.no_book))


if __name__ == "__main__":
    main()
