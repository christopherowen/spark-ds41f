#!/usr/bin/env python3
"""Start-up MemAvailable minimum of every boot in a promo_sampler.sh log.

A boot begins when MemAvailable falls from above --down GiB (the serving container
stopped); its start-up low is the minimum over the next --window seconds, which covers
weight loading, preparation, capture and the first minutes of serving. The table also
gives MemAvailable --settle seconds after the low, as the level start-up recovers to.
usage: boot_minima.py LOG [--down 90] [--window 300] [--settle 150]
"""
import argparse
import datetime as dt


def cet(epoch: float) -> str:
    # Central European Summer Time until the last Sunday of October (2026-10-25).
    offset = 2 if epoch < dt.datetime(2026, 10, 25, 1, tzinfo=dt.timezone.utc).timestamp() else 1
    return dt.datetime.fromtimestamp(epoch, dt.timezone(dt.timedelta(hours=offset))).strftime("%H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--down", type=float, default=90)
    parser.add_argument("--window", type=float, default=300)
    parser.add_argument("--settle", type=float, default=150)
    args = parser.parse_args()
    rows = []
    for line in open(args.log):
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            rows.append((float(parts[0]), int(parts[1]) / 2**20))
    starts, down = [], False
    for t, gib in rows:
        if gib > args.down:
            down = True
        elif down:
            starts.append(t)
            down = False
    print("boot  start CET  low GiB  low CET   after start  +settle GiB")
    for index, start in enumerate(starts, 1):
        window = [(t, g) for t, g in rows if start <= t <= start + args.window]
        if not window:
            continue
        low_t, low = min(window, key=lambda row: row[1])
        later = [g for t, g in rows if t >= low_t + args.settle]
        settle = f"{later[0]:.3f}" if later else "-"
        print(f"{index:>4}  {cet(start)}  {low:7.3f}  {cet(low_t)}  {low_t - start:9.1f} s  {settle:>10}")


if __name__ == "__main__":
    main()
