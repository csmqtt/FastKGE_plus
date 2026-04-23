"""Aggregate Whole_MRR / training-time stats across seeds.

Parses ./logs/multiseed/<tag>/s<seed>/<timestamp>/<dataset>.log files produced
by multiseed.sh and prints:

    1. Per-dataset Whole_MRR@S4 mean ± std, compared to the paper's Table 2/3
       numbers (Δ vs paper). This is the apples-to-apples comparison.
    2. Per-snapshot mean ± std, so you can see which snapshot dominates the
       variance (e.g. FACT S3/S4 are high-variance because loss-landscape is
       flat and patience=3 can stop early on bad seeds).
    3. Raw per-seed Whole_MRR@S4 table, so outliers are obvious.

Usage:
    python multiseed_collect.py <tag>
    python multiseed_collect.py <tag> --root ./logs/multiseed
    python multiseed_collect.py <tag> --compare <other_tag>
       -> side-by-side mean±std, with a Δ column (tag - other_tag) so you
          can tell at a glance whether your optimization actually beats the
          reference tag beyond the seed noise band.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
from collections import defaultdict
from typing import Any

REPORT_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$"
)
SUM_TIME = re.compile(r"Sum_Training_Time:([0-9.]+)")

# Paper Table 2/3 MRR (the "MRR" column; equivalent to Whole_MRR @ final snap).
PAPER_MRR: dict[str, float] = {
    "ENTITY":   0.239,
    "RELATION": 0.185,
    "FACT":     0.203,
    "HYBRID":   0.211,
    "FB_CKGE":  0.223,
    "WN_CKGE":  0.159,
}


def parse_log(path: str) -> dict[str, Any] | None:
    """Return {'snaps': {i: {time, mrr, h1, h3, h10}}, 'sum_time': float}.

    We take the LAST "Report Result" block in the file; main.py writes this
    block only once at the end of training so there's normally just one, but
    if a run was interrupted and resumed the last block is the canonical one.
    """
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    idx = text.rfind("Report Result")
    if idx < 0:
        return None
    tail = text[idx:]
    snaps: dict[int, dict[str, float]] = {}
    for line in tail.splitlines():
        m = REPORT_ROW.match(line)
        if m:
            snap = int(m.group(1))
            snaps[snap] = {
                "time": float(m.group(2)),
                "mrr":  float(m.group(3)),
                "h1":   float(m.group(4)),
                "h3":   float(m.group(5)),
                "h10":  float(m.group(6)),
            }
    sm = SUM_TIME.search(tail)
    return {"snaps": snaps, "sum_time": float(sm.group(1)) if sm else None}


def gather(tag_dir: str) -> dict[str, dict[int, dict[str, Any]]]:
    """dataset -> seed -> parsed; picks latest-timestamp log on collision."""
    by_ds: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for log_path in sorted(glob.glob(os.path.join(tag_dir, "s*", "*", "*.log"))):
        rel = os.path.relpath(log_path, tag_dir).split(os.sep)
        if len(rel) != 3 or not rel[0].startswith("s"):
            continue
        try:
            seed = int(rel[0][1:])
        except ValueError:
            continue
        ds = os.path.splitext(rel[2])[0]
        info = parse_log(log_path)
        if info is None or 4 not in info["snaps"]:
            print(f"[skip] incomplete log: {log_path}")
            continue
        prev = by_ds[ds].get(seed)
        if prev is None or rel[1] > prev["_ts"]:
            info["_ts"] = rel[1]
            info["_path"] = log_path
            by_ds[ds][seed] = info
    return by_ds


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.pstdev(xs)


def print_summary(tag: str, tag_dir: str, by_ds: dict[str, dict[int, dict[str, Any]]]) -> None:
    print(f"\n=========== tag = {tag}   ({tag_dir})   ===========\n")

    hdr = f"{'Dataset':<9} {'n':>2}  {'Whole_MRR@S4 mean ± std':<26} {'paper':>6}  {'Δ vs paper':>11}  {'T(s) mean':>10}"
    print(hdr)
    print("-" * len(hdr))
    for ds in sorted(by_ds.keys()):
        seeds = sorted(by_ds[ds].keys())
        mrrs = [by_ds[ds][s]["snaps"][4]["mrr"] for s in seeds]
        times = [by_ds[ds][s]["sum_time"] for s in seeds if by_ds[ds][s]["sum_time"] is not None]
        mu, sd = mean_std(mrrs)
        tmu = statistics.mean(times) if times else float("nan")
        paper = PAPER_MRR.get(ds)
        p_str = f"{paper:.3f}" if paper is not None else "  -  "
        d_str = f"{(mu - paper) * 100:+.2f} pp" if paper is not None else "    -    "
        print(f"{ds:<9} {len(seeds):>2}  {mu:.4f} ± {sd:.4f}            {p_str:>6}  {d_str:>11}  {tmu:>10.1f}")

    print("\nPer-snapshot Whole_MRR (mean ± std across seeds):")
    for ds in sorted(by_ds.keys()):
        seeds = sorted(by_ds[ds].keys())
        cells = []
        for snap in range(5):
            vals = [by_ds[ds][s]["snaps"][snap]["mrr"] for s in seeds if snap in by_ds[ds][s]["snaps"]]
            if not vals:
                cells.append(f"S{snap}=   -   ")
            else:
                mu, sd = mean_std(vals)
                cells.append(f"S{snap}={mu:.3f}±{sd:.3f}")
        print(f"  {ds:<9}  " + "  ".join(cells))

    print("\nRaw per-seed Whole_MRR@S4:")
    all_seeds = sorted({s for ds in by_ds.values() for s in ds.keys()})
    print(f"  {'Dataset':<9}  " + "  ".join(f"s{s:<6}" for s in all_seeds))
    for ds in sorted(by_ds.keys()):
        cells = [f"{by_ds[ds][s]['snaps'][4]['mrr']:.4f}" if s in by_ds[ds] else "   -  " for s in all_seeds]
        print(f"  {ds:<9}  " + "  ".join(f"{c:<7}" for c in cells))


def print_compare(
    tag_a: str, by_ds_a: dict[str, dict[int, dict[str, Any]]],
    tag_b: str, by_ds_b: dict[str, dict[int, dict[str, Any]]],
) -> None:
    print(f"\n=========== compare: {tag_a}  vs  {tag_b}  ===========\n")
    hdr = f"{'Dataset':<9}  {tag_a:<22}  {tag_b:<22}  {'Δ (A-B, pp)':>12}  {'signif?':>8}"
    print(hdr)
    print("-" * len(hdr))
    datasets = sorted(set(by_ds_a.keys()) | set(by_ds_b.keys()))
    for ds in datasets:
        mrrs_a = [by_ds_a[ds][s]["snaps"][4]["mrr"] for s in sorted(by_ds_a.get(ds, {}))]
        mrrs_b = [by_ds_b[ds][s]["snaps"][4]["mrr"] for s in sorted(by_ds_b.get(ds, {}))]
        if not mrrs_a or not mrrs_b:
            print(f"{ds:<9}  (missing in one tag)")
            continue
        mu_a, sd_a = mean_std(mrrs_a)
        mu_b, sd_b = mean_std(mrrs_b)
        delta_pp = (mu_a - mu_b) * 100
        # Rough significance proxy: |Δ| > sqrt(sd_a^2 + sd_b^2) (i.e. Δ > 1σ
        # of the difference distribution assuming independence). Not a formal
        # t-test, but good enough to flag "this is seed noise" vs "real".
        noise = ((sd_a ** 2 + sd_b ** 2) ** 0.5) * 100
        signif = "yes" if abs(delta_pp) > noise else "noise"
        str_a = f"{mu_a:.4f}±{sd_a:.4f} (n={len(mrrs_a)})"
        str_b = f"{mu_b:.4f}±{sd_b:.4f} (n={len(mrrs_b)})"
        print(f"{ds:<9}  {str_a:<22}  {str_b:<22}  {delta_pp:>+9.2f}    {signif:>8}")
    print("\n  'noise' means |Δ| < sqrt(σ_A² + σ_B²); treat as indistinguishable.")
    print("  'yes'   means |Δ| exceeds the combined 1σ band; likely a real effect,")
    print("          but for publication run a paired t-test on the per-seed pairs.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("tag", help="experiment tag under logs/multiseed/")
    ap.add_argument("--root", default="./logs/multiseed",
                    help="Root dir containing per-tag folders")
    ap.add_argument("--compare", default=None,
                    help="Another tag to diff against (shows per-dataset Δ ± signif)")
    args = ap.parse_args()

    tag_dir = os.path.join(args.root, args.tag)
    if not os.path.isdir(tag_dir):
        raise SystemExit(f"Not found: {tag_dir}")
    by_ds = gather(tag_dir)
    if not by_ds:
        raise SystemExit(f"No complete logs found under {tag_dir}")
    print_summary(args.tag, tag_dir, by_ds)

    if args.compare:
        other_dir = os.path.join(args.root, args.compare)
        if not os.path.isdir(other_dir):
            raise SystemExit(f"--compare tag not found: {other_dir}")
        by_ds_other = gather(other_dir)
        if not by_ds_other:
            raise SystemExit(f"No complete logs under {other_dir}")
        print_summary(args.compare, other_dir, by_ds_other)
        print_compare(args.tag, by_ds, args.compare, by_ds_other)


if __name__ == "__main__":
    main()
