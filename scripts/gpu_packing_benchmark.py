#!/usr/bin/env python
"""Throughput of N concurrent pmemd.cuda copies of one window on a single GPU, with and without MPS."""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

NS_DAY_RE = re.compile(r"ns/day\s*=\s*([0-9.]+)")
EPTOT_RE = re.compile(r"EPtot\s*=\s*(-?[0-9.]+|\*+|NaN)", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("window_dir", help="finished window directory (parm7, restraint.RST, start rst7, mdout)")
    p.add_argument("--prmtop", help="override the topology found in window_dir")
    p.add_argument("--inpcrd", help="override the start coordinates found in window_dir")
    p.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 12, 16])
    p.add_argument("--mps", nargs="+", choices=["off", "on"], default=["off", "on"])
    p.add_argument("--nstlim", type=int, default=125000, help="steps per copy (125000 = 500 ps at 4 fs)")
    p.add_argument("--ig", type=int, default=71277, help="fixed seed shared by every copy")
    p.add_argument("--device", default="0", help="CUDA device id every copy runs on")
    p.add_argument("--exe", default="pmemd.cuda")
    p.add_argument("--workdir", help="scratch for the runs (default: a temp dir under $TMPDIR)")
    p.add_argument("--csv", default="gpu_packing_benchmark.csv")
    p.add_argument("--sample-s", type=float, default=2.0, help="nvidia-smi sampling interval")
    return p.parse_args()


def find_inputs(window_dir, prmtop=None, inpcrd=None):
    """Topology, start coordinates, DISANG file and mdout of a finished window.

    Parameters
    ----------
    window_dir : str
        Window output directory.
    prmtop, inpcrd : str, optional
        Explicit paths that bypass the search.

    Returns
    -------
    tuple of str
        ``(prmtop, inpcrd, restraint, mdout)``.
    """
    def one(pattern, exclude=None):
        hits = [f for f in glob.glob(os.path.join(window_dir, pattern))
                if exclude is None or exclude not in os.path.basename(f)]
        if len(hits) != 1:
            sys.exit(f"expected one {pattern} in {window_dir}, found {hits}; pass it explicitly")
        return hits[0]

    prmtop = prmtop or one("*.parm7")
    inpcrd = inpcrd or one("*.rst7", exclude="restrt")
    restraint = os.path.join(window_dir, "restraint.RST")
    mdout = os.path.join(window_dir, "mdout")
    for f in (restraint, mdout):
        if not os.path.isfile(f):
            sys.exit(f"missing {f}")
    return prmtop, inpcrd, restraint, mdout


def mdin_from_mdout(mdout, nstlim, ig):
    """Rebuild the window's mdin from the echo in its mdout, with nstlim and ig overridden.

    Parameters
    ----------
    mdout : str
        mdout of the finished window.
    nstlim : int
        Steps per copy.
    ig : int
        Random seed.

    Returns
    -------
    str
        mdin text with ``DISANG = restraint.RST``.
    """
    lines, keep = [], False
    with open(mdout) as fh:
        for line in fh:
            if "Here is the input file:" in line:
                keep = True
                continue
            if keep and (line.startswith("Note:") or line.startswith("|") or "RESOURCE   USE" in line):
                break
            if keep:
                lines.append(line.rstrip())
    text = "\n".join(lines).strip("\n") + "\n"
    if "&cntrl" not in text:
        sys.exit(f"no mdin echo found in {mdout}")
    for key, value in (("nstlim", nstlim), ("ig", ig)):
        pattern = re.compile(rf"\b{key}\s*=\s*-?\d+", re.IGNORECASE)
        if not pattern.search(text):
            sys.exit(f"{key} not set in the echoed mdin")
        text = pattern.sub(f"{key} = {value}", text, count=1)
    return re.sub(r"DISANG\s*=\s*\S+", "DISANG = restraint.RST", text)


def gpu_info(device):
    out = subprocess.run(
        ["nvidia-smi", "-i", device, "--query-gpu=name,compute_mode,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True,
    )
    return out.stdout.strip()


def sample_gpu(device):
    out = subprocess.run(
        ["nvidia-smi", "-i", device, "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
    )
    try:
        util, mem = (float(x) for x in out.stdout.strip().split(","))
        return util, mem
    except ValueError:
        return None


def start_mps(root):
    """Start a private MPS daemon; returns its env, or None if it did not start."""
    env = dict(os.environ)
    env["CUDA_MPS_PIPE_DIRECTORY"] = os.path.join(root, "mps_pipe")
    env["CUDA_MPS_LOG_DIRECTORY"] = os.path.join(root, "mps_log")
    for d in (env["CUDA_MPS_PIPE_DIRECTORY"], env["CUDA_MPS_LOG_DIRECTORY"]):
        os.makedirs(d, exist_ok=True)
    r = subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[mps] could not start: {r.stderr.strip()}", flush=True)
        return None
    print(f"[mps] daemon started, pipe {env['CUDA_MPS_PIPE_DIRECTORY']}", flush=True)
    return env


def mps_server_running(env):
    r = subprocess.run(["nvidia-cuda-mps-control"], input="get_server_list\n", env=env,
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def stop_mps(env):
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", env=env, capture_output=True, text=True)
    time.sleep(2)


def read_mdout(path):
    """``(ns_per_day, eptot_series, ok)`` from one mdout; ns/day is the all-steps average."""
    if not os.path.isfile(path):
        return None, [], False
    with open(path, errors="replace") as fh:
        text = fh.read()
    body = text.split("A V E R A G E S")[0]
    eptot = EPTOT_RE.findall(body)
    ok = bool(eptot) and all(re.fullmatch(r"-?[0-9.]+", e) for e in eptot)
    ns_day = None
    tail = text.split("Average timings for all steps:")
    if len(tail) > 1:
        m = NS_DAY_RE.search(tail[1])
        ns_day = float(m.group(1)) if m else None
    return ns_day, [float(e) for e in eptot] if ok else [], ok and ns_day is not None


def run_cell(n, mps, inputs, mdin_text, args, root, env):
    """Run n copies at once on one device and return a result row."""
    prmtop, inpcrd, restraint, _ = inputs
    cell = os.path.join(root, f"mps_{mps}", f"n{n:02d}")
    procs = {}
    env = dict(env)
    env["CUDA_VISIBLE_DEVICES"] = args.device
    t0 = time.time()
    for i in range(n):
        d = os.path.join(cell, f"c{i:02d}")
        os.makedirs(d, exist_ok=True)
        for src, name in ((prmtop, "sys.parm7"), (inpcrd, "start.rst7"), (restraint, "restraint.RST")):
            shutil.copy(src, os.path.join(d, name))
        with open(os.path.join(d, "mdin"), "w") as fh:
            fh.write(mdin_text)
        cmd = [args.exe, "-O", "-i", "mdin", "-p", "sys.parm7", "-c", "start.rst7",
               "-o", "mdout", "-r", "restrt.rst7", "-x", "traj.nc", "-inf", "mdinfo"]
        log = open(os.path.join(d, "stdout.txt"), "w")
        p = subprocess.Popen(cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
        # keep the Popen alive: a collected one is reaped by subprocess, and wait4 then fails
        procs[p.pid] = (d, log, p)

    samples, cpu_s, codes, mps_seen = [], {}, {}, None
    pending = set(procs)
    while pending:
        s = sample_gpu(args.device)
        if s:
            samples.append(s)
        if mps == "on" and not mps_seen:
            mps_seen = mps_server_running(env)
        for pid in list(pending):
            done, status, ru = os.wait4(pid, os.WNOHANG)
            if done:
                pending.discard(pid)
                codes[pid] = os.waitstatus_to_exitcode(status)
                cpu_s[pid] = ru.ru_utime + ru.ru_stime
                procs[pid][1].close()
                procs[pid][2].returncode = codes[pid]
        if pending:
            time.sleep(args.sample_s)
    wall = time.time() - t0

    per_copy, series, ok = [], [], 0
    for pid, (d, _, _) in procs.items():
        ns_day, ep, good = read_mdout(os.path.join(d, "mdout"))
        good = good and codes[pid] == 0
        ok += good
        if good:
            per_copy.append(ns_day)
            series.append(ep)
        nc = os.path.join(d, "traj.nc")
        if os.path.exists(nc):
            os.remove(nc)

    sim_ns = args.nstlim * dt_ps(mdin_text) / 1000.0
    utils = [u for u, _ in samples]
    return dict(
        mps=mps,
        mps_active="" if mps == "off" else bool(mps_seen),
        n_copies=n,
        ok_copies=ok,
        wall_s=round(wall, 1),
        per_copy_ns_day_mean=round(sum(per_copy) / len(per_copy), 1) if per_copy else "",
        per_copy_ns_day_min=round(min(per_copy), 1) if per_copy else "",
        total_ns_day=round(ok * sim_ns / wall * 86400.0, 1),
        sum_ns_day=round(sum(per_copy), 1) if per_copy else "",
        gpu_util_mean=round(sum(utils) / len(utils), 1) if utils else "",
        gpu_mem_peak_mib=max(m for _, m in samples) if samples else "",
        cpu_frac_per_copy=round(sum(cpu_s.values()) / len(cpu_s) / wall, 2) if cpu_s else "",
    ), series


def dt_ps(mdin_text):
    m = re.search(r"\bdt\s*=\s*([0-9.]+)", mdin_text)
    return float(m.group(1)) if m else 0.001


def max_deviation(series, ref):
    """Largest |dEPtot| between any copy and the reference series, over their common records."""
    worst = 0.0
    for ep in series:
        k = min(len(ep), len(ref))
        if k:
            worst = max(worst, max(abs(a - b) for a, b in zip(ep[:k], ref[:k])))
    return worst


def main():
    args = parse_args()
    inputs = find_inputs(args.window_dir, args.prmtop, args.inpcrd)
    mdin_text = mdin_from_mdout(inputs[3], args.nstlim, args.ig)
    root = args.workdir or tempfile.mkdtemp(prefix="gpu_pack_", dir=os.environ.get("TMPDIR"))
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "mdin"), "w") as fh:
        fh.write(mdin_text)

    info = gpu_info(args.device)
    print(f"[gpu] device {args.device}: {info}", flush=True)
    print(f"[inputs] {inputs[0]}\n         {inputs[1]}\n[workdir] {root}", flush=True)

    exe = shutil.which(args.exe)
    if exe is None:
        sys.exit(f"{args.exe} not found on PATH (module load amber/25/gnu?)")
    if not os.access(exe, os.X_OK):
        sys.exit(f"{exe} is not executable")
    args.exe = exe
    print(f"[exe] {exe}", flush=True)

    rows, ref = [], None
    for mps in args.mps:
        env = dict(os.environ)
        if mps == "on":
            env = start_mps(root)
            if env is None:
                rows.append(dict(mps="on", mps_active=False, n_copies="", ok_copies=0))
                continue
        try:
            for n in args.copies:
                row, series = run_cell(n, mps, inputs, mdin_text, args, root, env)
                if ref is None and series:
                    ref = series[0]
                row["max_abs_dEPtot_vs_first"] = round(max_deviation(series, ref), 4) if ref else ""
                row["gpu"] = info
                rows.append(row)
                print(f"[cell] mps={mps:3s} n={n:2d} ok={row['ok_copies']:2d} wall={row['wall_s']:7.1f}s "
                      f"total={row['total_ns_day']:8.1f} sum={row['sum_ns_day']} ns/day "
                      f"per-copy={row['per_copy_ns_day_mean']} "
                      f"util={row['gpu_util_mean']}% cpu/copy={row['cpu_frac_per_copy']} "
                      f"dEPtot={row['max_abs_dEPtot_vs_first']}", flush=True)
                write_csv(args.csv, rows)
        finally:
            if mps == "on":
                stop_mps(env)

    write_csv(args.csv, rows)
    print(f"[done] {args.csv}", flush=True)


def write_csv(path, rows):
    """Write all rows so far; speedup is against the single-copy MPS-off cell, else any single copy."""
    singles = [r for r in rows if r.get("n_copies") == 1 and r.get("sum_ns_day")]
    base = next((r["sum_ns_day"] for r in singles if r["mps"] == "off"), None)
    base = base or (singles[0]["sum_ns_day"] if singles else None)
    for r in rows:
        if base and r.get("sum_ns_day"):
            r["speedup_vs_single"] = round(r["sum_ns_day"] / base, 2)
    fields = ["gpu", "mps", "mps_active", "n_copies", "ok_copies", "wall_s", "per_copy_ns_day_mean",
              "per_copy_ns_day_min", "total_ns_day", "sum_ns_day", "speedup_vs_single", "gpu_util_mean",
              "gpu_mem_peak_mib", "cpu_frac_per_copy", "max_abs_dEPtot_vs_first"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
