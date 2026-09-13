#!/usr/bin/env python
"""Parse [TIMING] records from an ISDDM/Toil run log into a per-phase wall-clock + compute-hours breakdown.

Each MD / post-analysis subprocess (simulations.Calculation.run) and each MBAR solve
(adaptive_restraints.compute_mbar) emits a line:

    [TIMING] phase=<phase> wall_s=<float> cores=<float> gpu=<0|1> end=<epoch_seconds> | <detail>

Phases:
  endstate        - endstate setup MD (minimization / basic-MD / REMD)
  adaptive        - the ALS pilot (its MD + post-analysis + band MBAR; detected by the *_pilot output tree)
  intermediate_md - production intermediate-window MD
  post_analysis   - production post-analysis re-scoring (sander imin=5, the N^2 grid)
  mbar            - production free-energy MBAR

Per phase it reports:
  jobs       - number of timed subprocess / MBAR jobs
  elapsed_h  - parallel-adjusted wall-clock span (max end - min start) of that phase's jobs
  cpu_h      - sum of wall*cores/3600 over CPU jobs (gpu=0)
  gpu_h      - sum of wall/3600 over GPU jobs (gpu=1; one GPU per window)

Usage:
    python timing_report.py <toil_log_file> [more_logs ...]
    python timing_report.py --per-job <phase> <toil_log_file> [more_logs ...]

--per-job <phase> dumps every job in that phase (wall_s, cores, cpu_h) sorted
slowest-first, with min/median/max and the tail's share of the phase total --
use it to see whether one or two windows gate the phase and to confirm the
cores= reservation each job actually carried.
"""
import re
import sys
from collections import defaultdict

_PAT = re.compile(
    r"\[TIMING\]\s+phase=(?P<phase>\w+)\s+wall_s=(?P<wall>[\d.]+)\s+"
    r"cores=(?P<cores>[\d.]+)\s+gpu=(?P<gpu>[01])\s+end=(?P<end>\d+)"
)
PHASE_ORDER = ["endstate", "adaptive", "intermediate_md", "post_analysis", "mbar"]


def parse(paths):
    """Return {phase: [(wall_s, cores, gpu, end_epoch), ...]} from one or more log files."""
    rows = defaultdict(list)
    for path in paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                m = _PAT.search(line)
                if m:
                    rows[m["phase"]].append(
                        (float(m["wall"]), float(m["cores"]), int(m["gpu"]), float(m["end"]))
                    )
    return rows


def summarize(rows):
    """Aggregate per-phase jobs / elapsed span / CPU-hours / GPU-hours."""
    per = {}
    for phase, recs in rows.items():
        starts = [e - w for w, c, g, e in recs]
        ends = [e for w, c, g, e in recs]
        per[phase] = {
            "jobs": len(recs),
            "elapsed_h": (max(ends) - min(starts)) / 3600.0 if recs else 0.0,
            "cpu_h": sum(w * c for w, c, g, e in recs if g == 0) / 3600.0,
            "gpu_h": sum(w for w, c, g, e in recs if g == 1) / 3600.0,
            "_starts": starts,
            "_ends": ends,
        }
    return per


def format_report(per):
    order = [p for p in PHASE_ORDER if p in per] + [p for p in per if p not in PHASE_ORDER]
    hdr = f"{'phase':<16}{'jobs':>8}{'elapsed_h':>12}{'cpu_h':>12}{'gpu_h':>12}"
    out = [hdr, "-" * len(hdr)]
    tot_cpu = tot_gpu = 0.0
    all_starts, all_ends = [], []
    for p in order:
        d = per[p]
        out.append(
            f"{p:<16}{d['jobs']:>8}{d['elapsed_h']:>12.3f}{d['cpu_h']:>12.3f}{d['gpu_h']:>12.3f}"
        )
        tot_cpu += d["cpu_h"]
        tot_gpu += d["gpu_h"]
        all_starts += d["_starts"]
        all_ends += d["_ends"]
    out.append("-" * len(hdr))
    total_elapsed = (max(all_ends) - min(all_starts)) / 3600.0 if all_ends else 0.0
    out.append(
        f"{'TOTAL':<16}{sum(d['jobs'] for d in per.values()):>8}"
        f"{total_elapsed:>12.3f}{tot_cpu:>12.3f}{tot_gpu:>12.3f}"
    )
    out.append("")
    out.append(
        "notes: elapsed_h = parallel-adjusted span (max end - min start) per phase; phase spans may "
        "overlap slightly at DAG barriers. cpu_h = sum wall*cores (gpu=0); gpu_h = sum wall (gpu=1, "
        "1 GPU/window). 'adaptive' bundles the pilot's MD + post-analysis + band MBAR."
    )
    return "\n".join(out)


def format_per_job(rows, phase):
    """Per-job breakdown for one phase: wall_s, cores, cpu_h, slowest-first."""
    recs = rows.get(phase)
    if not recs:
        avail = ", ".join(sorted(rows)) or "(none)"
        return f"No jobs for phase={phase!r}. Phases present: {avail}"
    # (cpu_h, wall_s, cores, gpu) per job; cpu_h is wall*cores for CPU jobs, wall for GPU.
    jobs = sorted(
        ((w / 3600.0 if g else w * c / 3600.0, w, c, g) for w, c, g, e in recs),
        reverse=True,
    )
    unit = "gpu_h" if all(g for *_, g in jobs) else "cpu_h"
    out = [f"phase={phase}  jobs={len(jobs)}", ""]
    out.append(f"{'#':>4}{'wall_s':>12}{'cores':>8}{unit:>12}")
    out.append("-" * 36)
    for i, (ch, w, c, g) in enumerate(jobs, 1):
        out.append(f"{i:>4}{w:>12.1f}{c:>8g}{ch:>12.3f}")
    out.append("-" * 36)
    hours = [j[0] for j in jobs]
    total = sum(hours)
    mid = sorted(hours)[len(hours) // 2]
    coresvals = sorted({f"{j[2]:g}" for j in jobs})
    out.append("")
    out.append(f"total {unit}: {total:.3f}   min/median/max job: "
               f"{hours[-1]:.3f} / {mid:.3f} / {hours[0]:.3f}")
    if total > 0:
        out.append(f"slowest job = {100 * hours[0] / total:.1f}% of phase; "
                   f"top 3 = {100 * sum(hours[:3]) / total:.1f}%; "
                   f"top 5 = {100 * sum(hours[:5]) / total:.1f}%")
    out.append(f"cores reservation seen on these jobs: {', '.join(coresvals)}")
    return "\n".join(out)


def _hms(hours):
    """Human-readable H h MM m SS s from a float hours value."""
    s = int(round(max(0.0, hours) * 3600))
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s"


def format_overlap(per):
    """Total wall-clock + the intermediate_md vs post_analysis OVERLAP verdict.

    The whole point of the merged MD->post pipeline: post-analysis should run WHILE intermediate MD
    is still going (backfilling idle cores), not wait for every MD job to finish. This block answers
    that in one number: how much of the MD phase the post phase overlapped, and how much post added
    to the total wall on top of MD. Positive overlap => barrier dissolved.
    """
    all_starts = [s for d in per.values() for s in d["_starts"]]
    all_ends = [e for d in per.values() for e in d["_ends"]]
    total_h = (max(all_ends) - min(all_starts)) / 3600.0 if all_ends else 0.0
    lines = ["", "wall-clock", "-" * 52, f"  total elapsed: {total_h:.3f} h  ({_hms(total_h)})"]

    md, po = per.get("intermediate_md"), per.get("post_analysis")
    if md and po and md["_starts"] and po["_starts"]:
        md_s, md_e = min(md["_starts"]), max(md["_ends"])
        po_s, po_e = min(po["_starts"]), max(po["_ends"])
        md_span = (md_e - md_s) / 3600.0
        overlap_h = (md_e - po_s) / 3600.0            # >0 => post started before MD ended
        posts_during = sum(1 for s in po["_starts"] if s < md_e)
        post_tail_h = max(0.0, (po_e - md_e)) / 3600.0  # wall post added beyond the MD phase
        lines += [
            "",
            "overlap: intermediate_md vs post_analysis",
            "-" * 52,
            f"  MD phase span:                 {md_span:.3f} h  ({_hms(md_span)})",
        ]
        if overlap_h >= 0:
            pct = 100.0 * overlap_h / md_span if md_span > 0 else 0.0
            lines += [
                f"  post started {overlap_h:.3f} h BEFORE MD ended   ({pct:.0f}% of the MD phase overlapped)",
                f"  post jobs running during MD:   {posts_during}/{po['jobs']}",
                f"  post added to wall beyond MD:  {post_tail_h:.3f} h  ({_hms(post_tail_h)})",
                "  => OVERLAP: post backfills CPUs while MD runs (barrier dissolved)",
            ]
        else:
            lines += [
                f"  post started {-overlap_h:.3f} h AFTER MD ended",
                "  => NO OVERLAP: post ran only after all MD finished",
                "     (barrier still present, OR MD too short/parallel to stagger -- e.g. CPU-only cb7)",
            ]
    return "\n".join(lines)


def main(argv):
    if len(argv) >= 2 and argv[1] == "--per-job":
        if len(argv) < 4:
            print("usage: timing_report.py --per-job <phase> <log> [more_logs ...]")
            return 1
        phase, logs = argv[2], argv[3:]
        rows = parse(logs)
        if not rows:
            print("No [TIMING] records found — run the instrumented build and check the log path.")
            return 1
        print(format_per_job(rows, phase))
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 1
    rows = parse(argv[1:])
    if not rows:
        print("No [TIMING] records found — run the instrumented build and check the log path.")
        return 1
    per = summarize(rows)
    print(format_report(per))
    print(format_overlap(per))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
