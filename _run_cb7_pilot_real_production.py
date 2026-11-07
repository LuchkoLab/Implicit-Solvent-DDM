"""Driver: REAL production DDM run (4 fs / 10 ns intermediate windows) with the ALS pilot enabled.

Uses implicit_solvent_ddm/tests/input_files/real_production_run.yaml (cb7-adenosine, HMR topologies,
igb=2, endstate_method=0 -> the provided _298K.nc are the endstate trajectories). The intermediate
mdin (four_fento_time_step_mdin_template.mdin) is dt=0.004 (4 fs), nstlim=2_500_000 (10 ns).

ALS PILOT (Phase 4.5, observational in Step 5a): with adaptive_lambda=True the workflow runs a 50 ps
pilot of the COMPLEX leg (50 ps / 0.004 = 12500 steps at 4 fs; ntwx auto-set for ~100 frames) and LOGS
the converged restraint schedule. IMPORTANT: in Step 5a this is OBSERVATIONAL — production Phases 5/6/7
still run on the SEED schedule [-8,-2,4] (3 restraint windows) at full 10 ns, so the free energies are
computed on the seed, NOT the ALS-reduced schedule. Use the pilot log to see what ALS would pick;
applying it to production is Step 5b.

What to look for (grep '[ALS]' on the logfile):
  - "[ALS][pilot] Phase 4.5 starting. pilot output dir: .../mdgb_pilot"     (output isolation)
  - "[ALS][complex] restraint band ... pair (con X <-> Y): overlap=Z [ok|WEAK]"
  - "[ALS][complex] inserting restraint window ..." then "[ALS][pilot] CONVERGED ... N windows"
Overlap matrices for all three legs are written to .cache/<complex>/<leg>_O_MBAR.{h5,pdf}.

Run on a cluster with AMBER + isddm_env:
    SCRATCH=./real_prod_run python _run_cb7_pilot_real_production.py

NOTE (production cost): the intermediate mdin writes a trajectory frame every ntwx steps; the CPU N^2
post-analysis re-scores EVERY frame under EVERY state, so the frame count is the dominant cost lever.
At nstlim=2_500_000 / ntwx=250 that is 10000 frames/window -> consider a larger ntwx for production.
"""
import os
import re

import yaml

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.implicit_ddm_workflow import ddm_workflow
from toil.common import Toil
from toil.job import Job

SCRATCH = os.environ.get("SCRATCH", os.path.abspath("./real_prod_run"))
os.makedirs(SCRATCH, exist_ok=True)

LOGFILE = os.environ.get("PILOT_LOGFILE", os.path.join(SCRATCH, "real_production.log"))

options = Job.Runner.getDefaultOptions(os.path.join(SCRATCH, "jobstore"))
options.logLevel = os.environ.get("TOIL_LOG", "INFO")      # INFO so the [ALS] logs show
options.logFile = LOGFILE                                   # full Toil log (incl. [ALS] lines) -> file
options.clean = "always"
options.workDir = SCRATCH
options.maxCores = 50

with open("implicit_solvent_ddm/tests/input_files/real_production_run.yaml") as fh:
    cfg_dict = yaml.safe_load(fh)

config = Config.from_config(cfg_dict)

# --- enable the ALS pilot (Phase 4.5) ---
config.workflow.adaptive_lambda = True
# NO pilot_nstlim override: the pilot defaults to pilot_ps=50 ps, derived from the mdin's dt (4 fs)
# -> 12500 steps, with ntwx auto-set for ~pilot_frames (100) frames. (Set config.intermediate_args.
# pilot_nstlim only to force an explicit step count, e.g. a tiny test system.)
config.workflow.plot_overlap_matrix = True   # export per-leg overlap PDFs (raw .h5 always written)

# Route every output onto the local scratch fs (also avoids macOS ._ AppleDouble on an sshfs mount).
config.system_settings.working_directory = SCRATCH
config.system_settings.cache_directory_output = SCRATCH
config.workflow.ignore_receptor_endstate = False

os.makedirs(config.system_settings.top_directory_path, exist_ok=True)

complex_name = re.sub(r"\..*", "", os.path.basename(config.endstate_files.complex_parameter_filename))

print(
    f"[driver] scratch={SCRATCH}  adaptive_lambda=True  pilot=50 ps (12500 steps @ 4 fs)\n"
    f"[driver] complex={complex_name}\n"
    f"[driver] logfile: {LOGFILE}\n"
    f"[driver] production tree: {config.system_settings.top_directory_path}  (SEED schedule, 10 ns)\n"
    f"[driver] pilot tree (expected): {config.system_settings.top_directory_path}_pilot\n"
    f"[driver] NOTE: Step 5a pilot is observational — production uses the SEED schedule, not the ALS one\n"
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
prod = config.system_settings.top_directory_path
pilot = prod + "_pilot"
cache = os.path.join(SCRATCH, ".cache", complex_name)
print(f"[driver] production dir exists: {os.path.isdir(prod)} ({prod})", flush=True)
print(f"[driver] pilot dir exists:      {os.path.isdir(pilot)} ({pilot})", flush=True)
print(f"[driver] full Toil log written to: {LOGFILE}", flush=True)
if os.path.isdir(cache):
    overlaps = sorted(f for f in os.listdir(cache) if "_O_MBAR." in f)
    print(f"[driver] exported overlap matrices in {cache}:\n  " + "\n  ".join(overlaps), flush=True)
print("[driver] grep the schedule with:  grep '\\[ALS\\]' " + LOGFILE, flush=True)
