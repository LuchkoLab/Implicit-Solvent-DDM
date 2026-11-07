"""Driver: run the cb7 DDM workflow with post-production adaptive lambda scheduling (ALS).

Production runs on the config's windows (round 0); then every adjacent pair whose two-state overlap
is below ``min_degree_overlap`` gets a window in its gap and the run re-scores with warm resume.

Environment:
    CONFIG      config YAML (default implicit_solvent_ddm/tests/input_files/config.yaml, 100-step windows)
    THRESHOLD   min_degree_overlap (default 0.5: every distinct pair counts as weak, so a smoke test
                inserts in every band deterministically despite the short windows)
    ROUNDS      als_max_rounds (default 1)
    IN_SOLVENT  1 (default) decouples in GB water (charge, LJ, screening bands); 0 keeps the gas cycle
    SCRATCH     output + jobstore directory (default ./cb7_als_run)

What to check afterwards:
    grep '\\[ALS\\]' $SCRATCH/cb7_als.log                    # rounds, audit, plans, stop reason
    $SCRATCH/.cache/<complex>/als/{history.csv,final.json,round_XX/pairs.csv}

Run on a cluster with AMBER + isddm_env (cb7 is CPU-only):
    SCRATCH=./cb7_als_run python scripts/_run_cb7_als.py      # from the repo root
"""
import os
import sys

# run from the repo root; import this checkout's package, not an installed copy
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import re

import yaml
from toil.common import Toil
from toil.job import Job

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.implicit_ddm_workflow import ddm_workflow

CONFIG = os.environ.get("CONFIG", "implicit_solvent_ddm/tests/input_files/config.yaml")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
ROUNDS = int(os.environ.get("ROUNDS", "1"))
IN_SOLVENT = os.environ.get("IN_SOLVENT", "1") == "1"
SCRATCH = os.path.abspath(os.environ.get("SCRATCH", "./cb7_als_run"))
os.makedirs(SCRATCH, exist_ok=True)
LOGFILE = os.path.join(SCRATCH, "cb7_als.log")

with open(CONFIG) as fh:
    cfg_dict = yaml.safe_load(fh)
cfg_dict.setdefault("workflow_jobs", {})["adaptive_lambda"] = True
cfg_dict["workflow_jobs"]["plot_overlap_matrix"] = True
args = cfg_dict["workflow"]["intermediate_states_arguments"]
args["min_degree_overlap"] = THRESHOLD
args["als_max_rounds"] = ROUNDS
if IN_SOLVENT:
    args["decouple_in_solvent"] = True
    args.setdefault("lj_scale_windows", [])
    args.setdefault("screening_windows", [])

config = Config.from_config(cfg_dict)
config.system_settings.working_directory = SCRATCH
config.system_settings.cache_directory_output = SCRATCH
config.workflow.ignore_receptor_endstate = False
os.makedirs(config.system_settings.top_directory_path, exist_ok=True)

options = Job.Runner.getDefaultOptions(os.path.join(SCRATCH, "jobstore"))
options.logLevel = os.environ.get("TOIL_LOG", "INFO")
options.logFile = LOGFILE
options.clean = "always"
options.workDir = SCRATCH

complex_name = re.sub(r"\..*", "", os.path.basename(config.endstate_files.complex_parameter_filename))
print(f"[driver] {CONFIG}: threshold={THRESHOLD} rounds={ROUNDS} in_solvent={IN_SOLVENT}\n"
      f"[driver] scratch={SCRATCH} log={LOGFILE}", flush=True)
with Toil(options) as toil:
    config.endstate_files.toil_import_parameters(toil=toil)
    config.intermediate_args.toil_import_user_mdin(toil=toil)
    config.inputs["min_mdin"] = str(
        toil.import_file("file://" + os.path.abspath("implicit_solvent_ddm/tests/input_files/min.mdin"))
    )
    toil.start(Job.wrapJobFn(ddm_workflow, config))

als_dir = os.path.join(SCRATCH, ".cache", complex_name, "als")
print(f"[driver] WORKFLOW DONE; ALS history in {als_dir}", flush=True)
if os.path.exists(os.path.join(als_dir, "history.csv")):
    print(open(os.path.join(als_dir, "history.csv")).read(), flush=True)
