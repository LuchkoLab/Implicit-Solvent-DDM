"""Driver: cb7 DDM on GPU+CPU — demonstrate the merged MD->post pipeline overlap.

This is the run that actually shows the bottleneck is gone. cb7 CPU-only runs can't: their MD is
sub-second, faster than Toil's per-job scheduling latency, so MD always finishes before any post can
launch (nothing to overlap). With CUDA=True the complex/receptor MD windows run on the GPU
(pmemd.cuda), limited to #GPUs at a time, so they STAGGER and each takes real wall-time (GPU init +
MD). As each GPU window finishes, its CPU post-analysis re-scores fire and run on the CPUs IN
PARALLEL with the next GPU MD wave -- exactly the overlap the barrier used to prevent.

It also exercises the H1 fix that CPU-only cb7 never touches: complex/receptor GPU MD now request
cores=0 (setup_simulations.py) so a freed GPU is never stalled behind long CPU post jobs draining the
core pool. Watch for: (a) Toil accepting the cores=0 request (if it errors on resources, revert to
0.1), and (b) GPU MD windows starting back-to-back as GPUs free while CPUs are busy with post.

By design (memory: complex/receptor -> GPU, ligand -> CPU): complex/receptor MD use `pmemd.cuda`;
ligand MD and ALL post-analysis (sander imin=5) stay on CPU. Needs a CPU `pmemd` binary for the
ligand leg + `sander` for post (a full AMBER GPU install has both).

Env knobs:
    MAXCORES   cap the CPU pool (e.g. 40).            default: unset (all cores)
    EXECUTABLE GPU MD binary.                          default: pmemd.cuda
    ADAPTIVE   "1" to also run the ALS pilot.          default: 0 (off -> clean Phase 5/6 signal)
    NSTLIM     lengthen intermediate MD (steps) so the GPU MD phase spans real wall-time and the
               MD/post overlap is unmistakable. Rewrites a scratch copy of the intermediate mdin;
               ntwx is auto-scaled to keep ~FRAMES frames so post-analysis cost stays fixed.
               default: unset (use the mdin's nstlim=100). NOTE cb7 is ~150 atoms so GPU MD is
               init-bound -- you likely need a LARGE nstlim (e.g. 250000-2000000) to hit ~10-30 s
               per window; bump it if windows are still sub-second.
    FRAMES     trajectory frames to keep when NSTLIM is set (ntwx = nstlim/FRAMES).  default: 10
    SCRATCH    output + jobstore dir.                  default: ./cb7_gpu_run
    TOIL_LOG   Toil log level.                         default: INFO (needed to capture [TIMING])

Run on a GPU node with AMBER (GPU build) + isddm_env:
    MAXCORES=40 SCRATCH=./cb7_gpu_run python _run_cb7_gpu.py

Then read the overlap with timing_report.py (post_analysis should START before intermediate_md ENDS):
    python timing_report.py ./cb7_gpu_run/cb7_gpu.log
    python timing_report.py --per-job intermediate_md ./cb7_gpu_run/cb7_gpu.log   # confirm gpu=1 windows
"""
import os

import yaml

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.implicit_ddm_workflow import (
    ddm_workflow,
    _confine_single_machine_to_allocated_gpus,
)
from toil.common import Toil
from toil.job import Job

SCRATCH = os.environ.get("SCRATCH", os.path.abspath("./cb7_gpu_run"))
os.makedirs(SCRATCH, exist_ok=True)

LOGFILE = os.environ.get("GPU_LOGFILE", os.path.join(SCRATCH, "cb7_gpu.log"))

options = Job.Runner.getDefaultOptions(os.path.join(SCRATCH, "jobstore"))
options.logLevel = os.environ.get("TOIL_LOG", "INFO")   # INFO so [TIMING]/[GPU] logToMaster lines land
options.logFile = LOGFILE                                # capture the full Toil log for timing_report.py
options.clean = "always"
options.workDir = SCRATCH
_maxcores = os.environ.get("MAXCORES")
if _maxcores:
    options.maxCores = int(_maxcores)

# Pin one GPU MD window per allocated GPU under single_machine (the same fix implicit_ddm_workflow.main
# applies). Without it every pmemd.cuda window lands on GPU 0 and the "staggered GPU waves" this test
# relies on collapse. Safe on a plain GPU node too (falls back to nvidia-smi to count devices).
_confine_single_machine_to_allocated_gpus(options)

with open("implicit_solvent_ddm/tests/input_files/config.yaml") as fh:
    cfg_dict = yaml.safe_load(fh)

config = Config.from_config(cfg_dict)

# --- GPU: complex/receptor MD -> pmemd.cuda; ligand MD + post stay CPU ---
config.system_settings.CUDA = True
config.system_settings.executable = os.environ.get("EXECUTABLE", "pmemd.cuda")
# num_accelerators auto-sets to 1 under CUDA in Config.__post_init__; set explicitly for clarity.
config.system_settings.num_accelerators = 1

# Pilot OFF by default so intermediate_md / post_analysis are the ONLY production phases (a clean
# overlap read). ADAPTIVE=1 adds the Phase 4.5 pilot back (its jobs show as phase=adaptive).
config.workflow.adaptive_lambda = os.environ.get("ADAPTIVE", "0") == "1"

# --- optional: lengthen intermediate MD so the GPU MD phase spans real wall-time and post/MD
# overlap is obvious. Rewrite nstlim (and ntwx, to keep ~FRAMES frames so post cost is unchanged)
# into a SCRATCH copy of the intermediate mdin, then point the config at it. ---
_nstlim = os.environ.get("NSTLIM")
if _nstlim:
    import re as _re

    _nst = int(_nstlim)
    _frames = max(1, int(os.environ.get("FRAMES", "10")))
    _ntwx = max(1, _nst // _frames)
    with open(config.intermediate_args.mdin_intermediate_file) as _fh:
        _txt = _fh.read()
    _txt = _re.sub(r"nstlim\s*=\s*\d+", f"nstlim = {_nst}", _txt)
    _txt = _re.sub(r"ntwx\s*=\s*\d+", f"ntwx = {_ntwx}", _txt)
    _dst = os.path.join(SCRATCH, "intermediate_long.mdin")
    with open(_dst, "w") as _fh:
        _fh.write(_txt)
    config.intermediate_args.mdin_intermediate_file = _dst
    print(
        f"[driver] lengthened intermediate MD: nstlim={_nst}, ntwx={_ntwx} "
        f"(~{_frames} frames) -> {_dst}",
        flush=True,
    )

# Route every output onto the local scratch fs.
config.system_settings.working_directory = SCRATCH
config.system_settings.cache_directory_output = SCRATCH
config.workflow.ignore_receptor_endstate = False

os.makedirs(config.system_settings.top_directory_path, exist_ok=True)

print(
    f"[driver] scratch={SCRATCH}  CUDA=True  executable={config.system_settings.executable}\n"
    f"[driver] maxCores={getattr(options, 'maxCores', None)}  adaptive_lambda={config.workflow.adaptive_lambda}\n"
    f"[driver] logfile: {LOGFILE}\n"
    f"[driver] production tree: {config.system_settings.top_directory_path}\n"
    f"[driver] complex/receptor MD -> GPU (cores=0); ligand MD + post-analysis -> CPU\n"
    f"[driver] starting Toil workflow ...",
    flush=True,
)
with Toil(options) as toil:
    config.endstate_files.toil_import_parameters(toil=toil)
    config.intermediate_args.toil_import_user_mdin(toil=toil)
    config.inputs["min_mdin"] = str(
        toil.import_file("file://" + os.path.abspath("implicit_solvent_ddm/tests/input_files/min.mdin"))
    )
    toil.start(Job.wrapJobFn(ddm_workflow, config))

print("[driver] WORKFLOW DONE", flush=True)
print(f"[driver] full Toil log written to: {LOGFILE}", flush=True)
print("[driver] read the overlap with:", flush=True)
print(f"[driver]   python timing_report.py {LOGFILE}", flush=True)
print("[driver]   -> confirm post_analysis START precedes intermediate_md END (barrier dissolved),", flush=True)
print("[driver]      and intermediate_md shows gpu_h > 0 (GPU MD ran).", flush=True)
