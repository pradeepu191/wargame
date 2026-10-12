# Running the recorder on an always-on machine

Six hours of top-of-book data gave one window; the markout-by-coin-and-hour intervals and the
per-wallet classification on marked fills need days. A laptop lid is the wrong tool. Any small
Linux VPS does it: 1 vCPU, 2 GB RAM, 20+ GB disk, roughly $4–6/month (Hetzner CX22, DigitalOcean
basic droplet, Vultr, etc.). Nothing here needs a key or an account on Hyperliquid — the public
WebSocket is read-only.

Disk: the four coins with trades + `l2Book` + `bbo` write roughly 100–200 MB/day of zstd parquet
(BTC's top of book changes ~8 times a second). A 20 GB disk holds two to three months; sync and
delete, or add `--no-book` on coins you only need trades for.

## Setup (Ubuntu 24.04, as root after the first login)
```bash
adduser --disabled-password --gecos "" wargame
apt-get update && apt-get install -y python3-venv git rsync
su - wargame
git clone https://github.com/pradeepu191/wargame.git ~/wargame
cd ~/wargame
python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install websockets pyarrow
.venv/bin/python data/record.py --coins BTC --hours 0.02 --out /tmp/rec_test   # ~1 min smoke test
exit
cp /home/wargame/wargame/data/record.service /etc/systemd/system/wargame-record.service
systemctl daemon-reload && systemctl enable --now wargame-record
journalctl -u wargame-record -f          # "subscribed to [...]" then "... N trades so far" every 5 min
```
The service restarts itself on any crash or reboot; closed hours are written every five minutes
and the open hour on stop, so a restart loses seconds, not hours. Stop with
`systemctl stop wargame-record`; update the code with `su - wargame -c "cd ~/wargame && git pull"`
then `systemctl restart wargame-record`.

## Getting the data to the Mac (or SuperCloud)
From the Mac, in the repo root (the parquet files land next to the laptop's own recordings; the
analysis reads them all):
```bash
rsync -avz --ignore-existing wargame@<vps-ip>:~/wargame/data/raw/live/ data/raw/live/
python analysis/calibrate.py --raw data/raw/live --out results/live_2026_10
```
`--ignore-existing` leaves hours already present alone, so a file the VPS is still appending to
(the current hour, on a reconnect) is not half-copied over a complete one. Run it daily or put
it in a `cron` line. `data/raw/` is gitignored; raw data never goes into the repository.

## Checks
* `python data/coverage.py --raw data/raw/live` — hours present, gaps, prints per hour.
* `ls -la data/raw/live | tail` on the VPS — one `trades_`, `l2_` and `bbo_` file per coin per
  hour, written a few minutes after the hour closes.
* `df -h` on the VPS once a week.
