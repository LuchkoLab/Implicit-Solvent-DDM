#!/usr/bin/env python
"""Prepare post-production ALS refinement of a finished in-solvent run, without rerunning REMD.

For each run (its config YAML) this:

1. checks the run is complete on disk (gate G0): result, windows, scored cells, REMD replicas;
2. rebuilds the complex and guest endstate trajectories at the target temperature from the REMD
   replicas, with the same cpptraj input the workflow used, and checks them against what the run
   scored (gate G1): frame count, the binding-mode frame, and the restraint energy of every frame
   under the strongest restraint window;
3. writes ``<W>/refine/config_files/<config>.yaml``: the original config plus
   ``run_endstate_method: False`` with the rebuilt trajectories (the endstate keeps its
   ``remd_endstate`` label, so every scored cell is reused), ``adaptive_lambda`` and its knobs, the
   original dG that round 0 must reproduce, and ``cache_directory_output: <W>/refine`` so the original
   results are never overwritten;
4. writes ``<W>/refine/ddm_submit/<guest>.sh`` pinned to a frozen code checkout, and a manifest.

Needs cpptraj on PATH (``module load amber/21/gnu``).

    python scripts/prepare_refine.py <config.yaml> ... --code <checkout> [--threshold 0.03]
    python scripts/prepare_refine.py --targets targets_t0.03.txt --code <checkout>
    python scripts/prepare_refine.py <config.yaml> --check-only     # G0 + G1, no config or script
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from als_census import schedule_from_yaml  # noqa: E402
from extend_weak_windows import band_restraints, read_traj, state_energies  # noqa: E402

# Keys the refine config may differ from the original in (dotted paths); anything else is a bug.
ALLOWED_CHANGES = {
    "workflow_jobs.run_endstate_method",
    "workflow_jobs.adaptive_lambda",
    "endstate_parameter_files.complex_endstate_trajectory",
    "endstate_parameter_files.ligand_endstate_trajectory",
    "endstate_parameter_files.receptor_parameter_filename",
    "endstate_parameter_files.receptor_coordinate_filename",
    "hardware_parameters.cache_directory_output",
    "workflow.intermediate_states_arguments.min_degree_overlap",
    "workflow.intermediate_states_arguments.als_max_rounds",
    "workflow.intermediate_states_arguments.als_max_new_windows_per_round",
    "workflow.intermediate_states_arguments.refine_expected_deltaG",
}


class GateError(RuntimeError):
    """A run failed a check and must not be refined as is."""


def stem(path):
    return re.sub(r"\..*", "", os.path.basename(str(path)))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def mdout_complete(path):
    """True if an AMBER mdout ran to its end (same marker as IntermidateRunner._check_mdout)."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 20000))
        return b"Final Performance Info" in fh.read()


# --------------------------------------------------------------------------------------------------
# What a run looks like on disk
# --------------------------------------------------------------------------------------------------
def run_paths(config_path):
    """Every path refinement needs, derived from the run's config."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    hw, files = cfg["hardware_parameters"], cfg["endstate_parameter_files"]
    out_name = str(hw["output_directory_name"])
    if out_name.endswith("/"):
        raise GateError(f"{config_path}: output_directory_name ends in '/', which changes where windows go")
    work = os.path.abspath(hw["working_directory"])
    top = os.path.join(work, out_name)
    complex_name = stem(files["complex_parameter_filename"])
    ligand_name = stem(files["ligand_parameter_filename"])
    stage1 = files.get("receptor_parameter_filename") is None
    receptor_name = "receptor_system" if stage1 else stem(files["receptor_parameter_filename"])
    guest = os.path.splitext(os.path.basename(config_path))[0].split("-", 1)[1]
    args = cfg["workflow"]["intermediate_states_arguments"]
    return dict(
        config=os.path.abspath(config_path), cfg=cfg, work=work, top=top, guest=guest, stage1=stage1,
        names={"complex": complex_name, "ligand": ligand_name, "receptor": receptor_name},
        parms={"complex": files["complex_parameter_filename"], "ligand": files["ligand_parameter_filename"],
               "receptor": files.get("receptor_parameter_filename")
               or os.path.join(top, "remd", "receptor_system", "receptor_system.parm7")},
        cache=os.path.join(os.path.abspath(hw["cache_directory_output"]), ".cache", complex_name),
        refine=os.path.join(work, "refine"),
        temperature=args.get("temperature", 298),
        submit=os.path.join(work, "ddm_submit", f"{guest}.sh"),
    )


def window_dirs(leg_dir):
    """Directories under a leg's window tree that hold an MD mdout."""
    found = []
    for root, dirs, files in os.walk(leg_dir):
        if "mdout" in files:
            found.append(root)
    return found


def count_cells(post_dir):
    n = 0
    for root, dirs, files in os.walk(post_dir):
        n += "simulation_mdout.parquet.gzip" in files
    return n


def inventory(p):
    """Gate G0: the finished run is complete and nothing it depends on is missing.

    Returns
    -------
    dict
        Facts recorded in the manifest.

    Raises
    ------
    GateError
    """
    problems, facts = [], {}
    dg_file = os.path.join(p["cache"], f"deltaG_{p['names']['complex']}.h5")
    if not os.path.exists(dg_file):
        raise GateError(f"no result {dg_file}")
    facts["deltaG"] = float(pd.read_hdf(dg_file)["deltaG"].iloc[0])
    _, cycle, _ = schedule_from_yaml(p["config"])
    for leg, name in p["names"].items():
        order = getattr(cycle, f"{leg}_order")
        windows = window_dirs(os.path.join(p["top"], name))
        incomplete = [w for w in windows if not mdout_complete(os.path.join(w, "mdout"))]
        bad_traj = [w for w in windows if len(glob.glob(os.path.join(w, "*.nc"))) != 1]
        if len(windows) != len(order) - 1:
            problems.append(f"{leg}: {len(windows)} window dirs, expected {len(order) - 1}")
        if incomplete:
            problems.append(f"{leg}: {len(incomplete)} incomplete mdout(s), e.g. {incomplete[0]}")
        if bad_traj:
            problems.append(f"{leg}: {len(bad_traj)} window(s) without exactly one .nc, e.g. {bad_traj[0]}")
        cells = count_cells(os.path.join(p["top"], "post_processing", name))
        expected = len(order) ** 2 + (1 if leg == "complex" else 0)
        if cells != expected:
            problems.append(f"{leg}: {cells} scored cells, expected {expected}")
        facts[f"{leg}_windows"], facts[f"{leg}_cells"] = len(windows), cells
    for leg in ("complex", "ligand") + (("receptor",) if p["stage1"] else ()):
        remd = os.path.join(p["top"], "remd", p["names"][leg])
        missing = [f"remd.nc.00{k}" for k in range(1, 9) if not os.path.exists(os.path.join(remd, f"remd.nc.00{k}"))]
        if missing:
            problems.append(f"{leg}: REMD replicas missing in {remd}: {missing}")
    if not os.path.exists(os.path.join(p["top"], "remd", p["names"]["complex"], "restraint.RST")):
        problems.append("complex REMD flat-bottom restraint.RST missing")
    frames = glob.glob(os.path.join(p["top"], p["names"]["complex"], "**", "*_lastframe.ncrst"), recursive=True)
    digests = {sha256(f) for f in frames}
    if len(digests) != 1:
        problems.append(f"{len(frames)} binding-mode frames with {len(digests)} different contents")
    else:
        facts["binding_mode"], facts["binding_mode_sha256"] = frames[0], digests.pop()
    for override in ("~/.mdgb.json", "~/.simrun.json"):
        if os.path.exists(os.path.expanduser(override)):
            problems.append(f"{override} overrides the directory structure")
    if problems:
        raise GateError("; ".join(problems))
    return facts


# --------------------------------------------------------------------------------------------------
# Endstate trajectories
# --------------------------------------------------------------------------------------------------
def rebuild_endstate(remd_dir, parm, temperature, out_nc):
    """Extract the target-temperature trajectory from REMD replicas, as ExtractTrajectories does."""
    replicas = [os.path.join(remd_dir, f"remd.nc.00{k}") for k in range(1, 9)]
    script = out_nc.replace(".nc", "_cpptraj.in")
    with open(script, "w") as fh:
        fh.write(f"parm {parm}\n"
                 f"trajin {replicas[0]} remdtraj remdtrajtemp {temperature} trajnames {','.join(replicas[1:])}\n"
                 f"trajout {out_nc} nobox\n\ngo\n")
    result = subprocess.run(["cpptraj", "-i", script], capture_output=True, text=True)
    with open(out_nc.replace(".nc", "_cpptraj.log"), "w") as fh:
        fh.write(result.stdout + result.stderr)
    if result.returncode != 0 or not os.path.exists(out_nc):
        raise GateError(f"cpptraj failed for {remd_dir}: see {out_nc.replace('.nc', '_cpptraj.log')}")


def endstate_cell(p, leg):
    """The endstate trajectory's scored cell under the leg's strongest restraint window, and that
    window's restraint file."""
    name = p["names"][leg]
    lambdas = [d for d in window_dirs(os.path.join(p["top"], name)) if f"{os.sep}lambda_window{os.sep}" in d]
    exponent = lambda d: float(d.split(f"lambda_window{os.sep}")[1].split(os.sep)[2])
    window = max(lambdas, key=exponent)
    tail = window.split(os.path.join(p["top"], name) + os.sep)[1]  # lambda_window/q/eps/con[/orient]
    pattern = os.path.join(p["top"], "post_processing", name, "remd_endstate", "*", "*", "*", "*" if leg != "complex" else "*/*",
                           tail, "*", "simulation_mdout.parquet.gzip")
    cells = glob.glob(pattern)
    if len(cells) != 1:
        raise GateError(f"{leg}: expected one endstate cell for {tail}, found {len(cells)}")
    return cells[0], os.path.join(window, "restraint.RST")


def check_endstate(p, leg, traj, facts, max_mismatched=0):
    """Gate G1: the rebuilt trajectory is the one the run scored.

    A frame matches when its restraint energy agrees with the scored one to 0.05 kcal/mol or 1e-4
    relative (under the strongest window these energies reach 1e5 kcal/mol). Up to ``max_mismatched``
    frames may differ; they are listed in the manifest.
    """
    xyz, _ = read_traj(traj)
    cell, rst = endstate_cell(p, leg)
    scored = pd.read_parquet(cell).sort_values("FRAME")
    if len(xyz) != len(scored):
        raise GateError(f"{leg}: rebuilt trajectory has {len(xyz)} frames, the run scored {len(scored)}")
    energy = state_energies(xyz, band_restraints([rst]))[:, 0]
    ref = scored.RESTRAINT.to_numpy()
    diff = np.abs(energy - ref)
    bad = diff > np.maximum(0.05, 1e-4 * np.abs(ref))
    good = diff[~bad]
    facts[f"{leg}_restraint_max_abs_diff"] = float(good.max())
    facts[f"{leg}_restraint_mean_abs_diff"] = float(good.mean())
    facts[f"{leg}_n_mismatched_frames"] = int(bad.sum())
    facts[f"{leg}_mismatched_frames"] = [dict(frame=int(i), scored=float(ref[i]), rebuilt=float(energy[i]))
                                         for i in np.flatnonzero(bad)]
    if bad.sum() > max_mismatched or good.mean() > 0.005:
        worst = int(np.argmax(diff))
        raise GateError(f"{leg}: {bad.sum()} rebuilt frames differ from the scored ones (allowed {max_mismatched}; "
                        f"worst frame {worst}: scored {ref[worst]:.4f}, rebuilt {energy[worst]:.4f} kcal/mol; "
                        f"mean over matching frames {good.mean():.5f})")
    if leg == "complex":
        reference, _ = read_traj(facts["binding_mode"])
        if reference.ndim == 2:  # a restart holds one frame: (atoms, 3)
            reference = reference[None]
        shift = float(np.abs(reference[-1] - xyz[-1]).max())
        facts["binding_mode_max_shift"] = shift
        if shift > 1e-3:
            raise GateError(f"complex: rebuilt last frame differs from the binding mode by {shift:.2e} A")


# --------------------------------------------------------------------------------------------------
# Refine config, submit script, manifest
# --------------------------------------------------------------------------------------------------
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def refine_config(p, trajs, facts, args):
    cfg = copy.deepcopy(p["cfg"])
    cfg.setdefault("workflow_jobs", {}).update(run_endstate_method=False, adaptive_lambda=True)
    files = cfg["endstate_parameter_files"]
    files["complex_endstate_trajectory"] = trajs["complex"]
    files["ligand_endstate_trajectory"] = trajs["ligand"]
    if p["stage1"]:
        files["receptor_parameter_filename"] = p["parms"]["receptor"]
        files["receptor_coordinate_filename"] = trajs["receptor"]
    cfg["hardware_parameters"]["cache_directory_output"] = p["refine"]
    knobs = cfg["workflow"]["intermediate_states_arguments"]
    knobs.update(min_degree_overlap=float(args.threshold), als_max_rounds=int(args.rounds),
                 refine_expected_deltaG=facts["deltaG"])
    if args.max_new_windows is not None:
        knobs["als_max_new_windows_per_round"] = int(args.max_new_windows)
    old, new = flatten(p["cfg"]), flatten(cfg)
    changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
    if changed - ALLOWED_CHANGES:
        raise GateError(f"refine config changes keys it must not: {sorted(changed - ALLOWED_CHANGES)}")
    return cfg


def submit_script(p, config_path, code, sha, tag):
    with open(p["submit"]) as fh:
        text = fh.read()
    text = re.sub(r"#SBATCH --job-name=\S+", f"#SBATCH --job-name=rf_{tag}", text)
    text = re.sub(r"#SBATCH --time=\S+", "#SBATCH --time=48:00:00", text)
    text = re.sub(r"--config_file \S+", f"--config_file {config_path}", text)
    text = re.sub(r"(file:/tmp/slurm\.\$SLURM_JOB_ID/)\S+", rf"\1ddm_job_store_rf_{p['guest']}", text)
    pin = (
        f"\n# refinement runs a frozen checkout, never the conda env's editable install\n"
        f"export PYTHONPATH={code}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        # from /, so the submit directory (python -c puts it first on sys.path) cannot shadow the pin
        f"(cd / && python -c \"import implicit_solvent_ddm.config as c; assert c.__file__.startswith('{code}/'), c.__file__; "
        f"assert 'complex_endstate_trajectory' in c.ParameterFiles.__dataclass_fields__\") "
        f"|| {{ echo 'ERROR: not running the frozen refine checkout'; exit 1; }}\n"
        f"test \"$(git -C {code} rev-parse HEAD)\" = \"{sha}\" || {{ echo 'ERROR: refine checkout moved'; exit 1; }}\n"
    )
    anchor = "export PYTHONNOUSERSITE=1\n"
    if anchor not in text:
        raise GateError(f"{p['submit']}: no '{anchor.strip()}' line to pin the code after")
    text = text.replace(anchor, anchor + pin, 1)
    run = "\"$CONDA_PREFIX/bin/run_implicit_ddm.py\""
    if run not in text:
        raise GateError(f"{p['submit']}: no run_implicit_ddm.py line")
    text = text.replace(run, "( while sleep 600; do du -sh /tmp/slurm.$SLURM_JOB_ID; done ) &\n" + run, 1)
    return text


def prepare(config_path, args):
    t0 = time.time()
    p = run_paths(config_path)
    facts = inventory(p)
    out = os.path.join(p["refine"], "endstates", p["names"]["complex"])
    os.makedirs(out, exist_ok=True)
    temperature = int(p["temperature"]) if float(p["temperature"]).is_integer() else p["temperature"]
    trajs = {}
    legs = ("complex", "ligand") + (("receptor",) if p["stage1"] else ())
    for leg in legs:
        trajs[leg] = os.path.join(out, f"{p['names'][leg]}_{temperature}K.nc")
        if args.force or not os.path.exists(trajs[leg]):
            rebuild_endstate(os.path.join(p["top"], "remd", p["names"][leg]), p["parms"][leg], temperature, trajs[leg])
        check_endstate(p, leg, trajs[leg], facts, args.max_mismatched_frames) if leg != "receptor" else None
        facts[f"{leg}_endstate_sha256"] = sha256(trajs[leg])
    manifest = dict(original_config=p["config"], original_config_sha256=sha256(p["config"]),
                    stage1=p["stage1"], endstates=trajs, checks=facts, check_seconds=round(time.time() - t0, 1))
    if args.check_only:
        return manifest
    sha = subprocess.run(["git", "-C", args.code, "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    cfg = refine_config(p, trajs, facts, args)
    os.makedirs(os.path.join(p["refine"], "config_files"), exist_ok=True)
    config_out = os.path.join(p["refine"], "config_files", os.path.basename(p["config"]))
    if os.path.exists(config_out) and not args.force:
        with open(config_out) as fh:
            if yaml.safe_load(fh) != cfg:
                raise GateError(f"{config_out} exists with different content; pass --force to replace it")
    with open(config_out, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    parts = p["config"].split(os.sep)
    tag = f"{parts[-6].replace('igb_', 'g')}{parts[-5]}{parts[-3].replace('rep', 'r')}_{p['guest']}"
    os.makedirs(os.path.join(p["refine"], "ddm_submit"), exist_ok=True)
    script = os.path.join(p["refine"], "ddm_submit", f"{p['guest']}.sh")
    with open(script, "w") as fh:
        fh.write(submit_script(p, config_out, os.path.abspath(args.code), sha, tag))
    manifest.update(refine_config=config_out, submit=script, code=os.path.abspath(args.code), code_sha=sha,
                    threshold=args.threshold, rounds=args.rounds, max_new_windows=args.max_new_windows,
                    schedule=json.loads(json.dumps(cfg["workflow"]["intermediate_states_arguments"], default=str)))
    os.makedirs(os.path.join(p["refine"], "manifests"), exist_ok=True)
    with open(os.path.join(p["refine"], "manifests", f"{p['names']['complex']}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", help="original run configs")
    ap.add_argument("--targets", help="file with one original config per line")
    ap.add_argument("--code", help="frozen checkout the refine jobs import (required unless --check-only)")
    ap.add_argument("--threshold", type=float, default=0.03, help="min_degree_overlap (two-state)")
    ap.add_argument("--rounds", type=int, default=3, help="als_max_rounds")
    ap.add_argument("--max-new-windows", type=int, default=None, help="als_max_new_windows_per_round")
    ap.add_argument("--max-mismatched-frames", type=int, default=0,
                    help="endstate frames whose restraint energy may differ from the scored one (G1)")
    ap.add_argument("--check-only", action="store_true", help="gates G0 and G1 only")
    ap.add_argument("--force", action="store_true", help="rebuild trajectories and replace refine configs")
    ap.add_argument("--report", help="CSV with one row per run")
    args = ap.parse_args()
    configs = list(args.configs)
    if args.targets:
        with open(args.targets) as fh:
            configs += [line.strip() for line in fh if line.strip()]
    if not configs:
        ap.error("no configs given")
    if not args.check_only and not args.code:
        ap.error("--code is required to write refine configs")
    rows = []
    for path in configs:
        try:
            m = prepare(path, args)
            rows.append(dict(config=path, status="ok", **{k: v for k, v in m["checks"].items()
                                                          if not isinstance(v, (dict, list))}))
            print(f"ok    {path}", flush=True)
        except (GateError, FileNotFoundError, subprocess.CalledProcessError) as err:
            rows.append(dict(config=path, status="gate_failed", detail=str(err)))
            print(f"FAIL  {path}: {err}", flush=True)
    if args.report:
        pd.DataFrame(rows).to_csv(args.report, index=False)
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
