#!/usr/bin/env python
"""Replica-exchange mixing report from an AMBER ``rem.log``.

Two numbers decide whether a T-REMD ladder is doing its job, and neither is the
per-pair acceptance rate on its own:

* **acceptance per adjacent pair** -- where the ladder is pinched. A rung pair below
  ~15% is a bottleneck; above ~60% the two rungs are redundant and the ladder is
  wasting ranks.
* **round trips** -- how often a replica actually walks from the bottom rung to the top
  and back. This is what buys decorrelation at the target temperature. A ladder can
  show 40% acceptance everywhere and still deliver zero round trips if one pair is
  pinched, so acceptance alone can look healthy while the ensemble at 298 K is a single
  trajectory wearing eight hats.

Occupancy is the third check: every replica should spend ~1/N of its time on every
rung. A replica parked on one rung is not sampling, whatever the acceptance says.

Usage::

    python scripts/remd_mixing.py path/to/rem.log
    python scripts/remd_mixing.py path/to/rem.log --ps-per-exchange 2.0 --csv mixing.csv

``--ps-per-exchange`` is ``nstlim * dt`` from the REMD mdin (500 * 0.004 = 2 ps for the
benchmark ladder); it only converts round-trip counts into a time, nothing else.
"""
import argparse
import os

import numpy as np


def parse_remlog(path):
    """Read an AMBER ``rem.log`` into per-exchange rung indices and energies.

    Parameters
    ----------
    path : str
        Path to ``rem.log``.

    Returns
    -------
    dict
        ``temps`` (n_rungs,) the sorted ladder, ``rung`` and ``new_rung``
        (n_exchange, n_replica) integer rung index of each replica before and after each
        attempt, and ``eptot`` (n_exchange, n_replica) potential energies.
    """
    before, after, energy = [], [], []
    block_t, block_new, block_e = [], [], []

    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if "exchange" in line and block_t:
                    before.append(block_t)
                    after.append(block_new)
                    energy.append(block_e)
                    block_t, block_new, block_e = [], [], []
                continue
            cols = line.split()
            if len(cols) < 7:
                continue
            block_e.append(float(cols[3]))
            block_t.append(float(cols[4]))
            block_new.append(float(cols[5]))

    if block_t:
        before.append(block_t)
        after.append(block_new)
        energy.append(block_e)

    # a killed run can leave a truncated final block
    width = len(before[0])
    keep = [i for i, b in enumerate(before) if len(b) == width]
    before = np.array([before[i] for i in keep])
    after = np.array([after[i] for i in keep])
    energy = np.array([energy[i] for i in keep])

    temps = np.unique(before)
    rung = np.searchsorted(temps, before)
    new_rung = np.searchsorted(temps, after)
    return dict(temps=temps, rung=rung, new_rung=new_rung, eptot=energy)


def attempted_parity(rung, new_rung, n_rungs):
    """Which pair set each exchange step attempted, as a 0/1 parity per step.

    AMBER alternates even pairs (0-1, 2-3, ...) with odd (1-2, 3-4, ...). The log records
    only accepted swaps, so the parity is read off the steps that swapped.

    Parameters
    ----------
    rung, new_rung : numpy.ndarray
        ``(n_exchange, n_replica)`` rung indices before and after each attempt.
    n_rungs : int
        Number of temperature rungs.

    Returns
    -------
    tuple of (numpy.ndarray, int)
        Parity of every step, and the number of steps whose observed swaps contradict
        it (0 for a well-formed log).
    """
    moved = new_rung != rung
    low = np.minimum(rung, new_rung)
    votes = np.zeros(2, dtype=int)

    for step in range(len(rung)):
        pairs = np.unique(low[step][moved[step]])
        if pairs.size == 0:
            continue
        votes[(pairs[0] - step) % 2] += 1

    offset = int(np.argmax(votes))
    parity = (np.arange(len(rung)) + offset) % 2

    contradictions = 0
    for step in range(len(rung)):
        pairs = np.unique(low[step][moved[step]])
        if pairs.size and np.any(pairs % 2 != parity[step]):
            contradictions += 1
    return parity, contradictions


def pair_acceptance(rung, new_rung, n_rungs):
    """Attempts, accepted swaps and acceptance rate for every adjacent rung pair.

    Parameters
    ----------
    rung, new_rung : numpy.ndarray
        ``(n_exchange, n_replica)`` rung indices before and after each attempt.
    n_rungs : int
        Number of temperature rungs.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, int)
        Attempts and accepted counts, both ``(n_rungs - 1,)``, and the number of steps
        that contradicted the inferred attempt pattern.
    """
    parity, contradictions = attempted_parity(rung, new_rung, n_rungs)
    moved = new_rung != rung
    low = np.minimum(rung, new_rung)

    accepted = np.zeros(n_rungs - 1, dtype=int)
    for step in range(len(rung)):
        pairs = low[step][moved[step]]
        for p in np.unique(pairs):
            accepted[p] += 1  # both partners report the swap; count the pair once

    pairs = np.arange(n_rungs - 1)
    attempts = np.array([np.sum(parity == (p % 2)) for p in pairs])
    return attempts, accepted, contradictions


def round_trips(rung, n_rungs):
    """Bottom -> top -> bottom traversals for each replica.

    One trip is one fresh configuration delivered to the target temperature.

    Parameters
    ----------
    rung : numpy.ndarray
        ``(n_exchange, n_replica)`` rung index of each replica.
    n_rungs : int
        Number of temperature rungs.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Completed round trips per replica, and the number of rungs each replica visited.
    """
    n_replica = rung.shape[1]
    trips = np.zeros(n_replica, dtype=int)
    visited = np.zeros(n_replica, dtype=int)

    for r in range(n_replica):
        track = rung[:, r]
        visited[r] = np.unique(track).size
        last_end = None
        for value in track:
            if value == 0:
                if last_end == "top":
                    trips[r] += 1
                last_end = "bottom"
            elif value == n_rungs - 1:
                last_end = "top"
    return trips, visited


def residence(rung, n_rungs):
    """How long a replica sits on a rung before the ladder moves it on.

    A diffusive walk leaves after ~1/acceptance attempts; far longer means the rung traps
    whatever lands on it.

    Parameters
    ----------
    rung : numpy.ndarray
        ``(n_exchange, n_replica)`` rung index of each replica.
    n_rungs : int
        Number of temperature rungs.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Mean and longest uninterrupted stay on each rung, in exchange attempts.
    """
    stays = [[] for _ in range(n_rungs)]
    for r in range(rung.shape[1]):
        track = rung[:, r]
        edges = np.flatnonzero(np.diff(track)) + 1
        for start, stop in zip(np.r_[0, edges], np.r_[edges, len(track)]):
            stays[track[start]].append(stop - start)

    mean = np.array([np.mean(s) if s else 0.0 for s in stays])
    longest = np.array([max(s) if s else 0 for s in stays])
    return mean, longest


def occupancy(rung, n_rungs):
    """Fraction of exchanges each replica spent on each rung, ``(n_replica, n_rungs)``."""
    n_replica = rung.shape[1]
    table = np.zeros((n_replica, n_rungs))
    for r in range(n_replica):
        counts = np.bincount(rung[:, r], minlength=n_rungs)
        table[r] = counts / counts.sum()
    return table


def report(data, ps_per_exchange=None):
    """Format the whole mixing report as one string."""
    temps = data["temps"]
    rung, new_rung, eptot = data["rung"], data["new_rung"], data["eptot"]
    n_rungs = temps.size
    n_exchange, n_replica = rung.shape

    attempts, accepted, contradictions = pair_acceptance(rung, new_rung, n_rungs)
    rates = accepted / np.maximum(attempts, 1)
    trips, visited = round_trips(rung, n_rungs)
    table = occupancy(rung, n_rungs)

    # mean energy per rung: a pinched pair is a gap wider than the fluctuations
    rung_energy = np.zeros(n_rungs)
    rung_sigma = np.zeros(n_rungs)
    for k in range(n_rungs):
        sel = eptot[rung == k]
        rung_energy[k] = sel.mean()
        rung_sigma[k] = sel.std()

    out = []
    out.append(f"exchanges parsed : {n_exchange}")
    out.append(f"replicas         : {n_replica}   rungs: {n_rungs}")
    if ps_per_exchange:
        out.append(
            f"simulated        : {n_exchange * ps_per_exchange / 1000:.2f} ns per replica "
            f"({ps_per_exchange:g} ps between attempts)"
        )
    if contradictions:
        out.append(f"WARNING: {contradictions} steps contradict the even/odd attempt pattern")
    out.append("")

    out.append("acceptance by adjacent pair")
    out.append(f"  {'pair':<18} {'attempts':>9} {'accepted':>9} {'rate':>7}   {'dE/sigma':>8}")
    for p in range(n_rungs - 1):
        gap = rung_energy[p + 1] - rung_energy[p]
        sigma = np.hypot(rung_sigma[p], rung_sigma[p + 1])
        flag = ""
        if rates[p] < 0.15:
            flag = "  <-- bottleneck"
        elif rates[p] > 0.60:
            flag = "  <-- redundant rung"
        out.append(
            f"  {temps[p]:6.2f} - {temps[p + 1]:6.2f} {attempts[p]:9d} {accepted[p]:9d} "
            f"{rates[p]:7.3f}   {gap / sigma:8.2f}{flag}"
        )
    out.append(f"  mean acceptance: {rates.mean():.3f}")
    out.append("")

    out.append("round trips (bottom rung -> top -> bottom)")
    for r in range(n_replica):
        out.append(
            f"  replica {r + 1}: {trips[r]:4d} trips   visited {visited[r]}/{n_rungs} rungs"
        )
    total = trips.sum()
    out.append(f"  total: {total}")
    if total:
        per_trip = n_exchange * n_replica / total
        out.append(f"  mean exchanges per round trip: {per_trip:.0f}")
        if ps_per_exchange:
            out.append(
                f"  mean time per round trip     : {per_trip * ps_per_exchange / 1000:.2f} ns"
            )
    out.append("")

    half = n_exchange // 2
    first, _ = round_trips(rung[:half], n_rungs)
    second, _ = round_trips(rung[half:], n_rungs)
    out.append(f"  first half: {first.sum()}   second half: {second.sum()}")
    out.append("")

    mean_stay, longest_stay = residence(rung, n_rungs)
    out.append("residence per rung (exchange attempts before moving on)")
    out.append(f"  {'rung':>8} {'mean':>8} {'longest':>9}   {'1/rate':>7}")
    for k in range(n_rungs):
        neighbours = rates[max(k - 1, 0):k + 1]
        expected = 1.0 / max(neighbours.mean(), 1e-9)
        out.append(
            f"  {temps[k]:8.2f} {mean_stay[k]:8.1f} {longest_stay[k]:9d}   {expected:7.1f}"
        )
    out.append("")

    out.append(f"rung occupancy (ideal {1 / n_rungs:.3f} everywhere)")
    header = "  replica " + " ".join(f"{t:7.1f}" for t in temps)
    out.append(header)
    for r in range(n_replica):
        out.append(f"  {r + 1:>7}  " + " ".join(f"{v:7.3f}" for v in table[r]))
    worst = np.abs(table - 1 / n_rungs).max()
    out.append(f"  largest deviation from uniform: {worst:.3f}")

    return "\n".join(out)


def write_csv(path, data):
    """Per-pair acceptance and per-replica round trips, as one flat CSV."""
    temps = data["temps"]
    n_rungs = temps.size
    attempts, accepted, _ = pair_acceptance(data["rung"], data["new_rung"], n_rungs)
    trips, visited = round_trips(data["rung"], n_rungs)

    with open(path, "w") as fh:
        fh.write("kind,label,attempts,accepted,rate\n")
        for p in range(n_rungs - 1):
            fh.write(
                f"pair,{temps[p]:.2f}-{temps[p + 1]:.2f},{attempts[p]},{accepted[p]},"
                f"{accepted[p] / max(attempts[p], 1):.4f}\n"
            )
        for r in range(len(trips)):
            fh.write(f"replica,{r + 1},,{trips[r]},{visited[r]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("remlog", help="path to rem.log")
    parser.add_argument(
        "--ps-per-exchange",
        type=float,
        default=None,
        help="nstlim * dt from the REMD mdin, to report round trips in ns",
    )
    parser.add_argument("--csv", default=None, help="also write the numbers here")
    args = parser.parse_args()

    data = parse_remlog(args.remlog)
    print(f"# {os.path.abspath(args.remlog)}")
    print(report(data, ps_per_exchange=args.ps_per_exchange))
    if args.csv:
        write_csv(args.csv, data)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
