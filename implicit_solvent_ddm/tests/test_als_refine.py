"""Unit tests for post-production ALS (``implicit_solvent_ddm.als_refine``).

Pure: no Toil, no MD. Run from the repository root::

    pytest implicit_solvent_ddm/tests/test_als_refine.py -v
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

import implicit_solvent_ddm.pandasmbar as pdmbar
from implicit_solvent_ddm import als_refine as ar
from implicit_solvent_ddm.matrix_order import CycleSteps, cycle_steps_from_config

INDEX_NAMES = ["traj_state", "traj_extdiel", "traj_charge", "traj_restraints"]
ENDSTATE = "endstate"


# --------------------------------------------------------------------------------------------------
# Synthetic energies: harmonic states u_j(x) = (x - mu_j)^2 / 2, samples x ~ N(mu_k, 1)
# --------------------------------------------------------------------------------------------------
def _harmonic(means, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    states = [("lambda_window", "78.5", "1.0", f"{k}.0") for k in range(len(means))]
    xs, rows = [], []
    for state, mu in zip(states, means):
        x = rng.normal(mu, 1.0, n)
        xs.append(x)
        rows += [state] * n
    x = np.concatenate(xs)
    u = np.column_stack([0.5 * (x - mu) ** 2 for mu in means])
    # flat tuple columns, as compute_mbar builds them
    return pd.DataFrame(u, index=pd.MultiIndex.from_tuples(rows, names=INDEX_NAMES),
                        columns=pd.Index(states, tupleize_cols=False))


def test_pairwise_overlap_is_not_diluted_by_neighbours():
    df = _harmonic(np.linspace(0.0, 0.1, 10))
    pairs = ar.adjacent_pair_overlaps(df)
    assert (pairs.status == "ok").all()
    assert pairs.o_pair.min() > 0.45  # near-identical neighbours: two-state overlap ~ 0.5
    full = pdmbar.mbar(df)[-1].compute_overlap()["matrix"]
    assert max(full[i][i + 1] for i in range(9)) < 0.15  # the full-N superdiagonal is ~ 1/K


def test_separated_pair_is_weak():
    pairs = ar.adjacent_pair_overlaps(_harmonic([0.0, 0.5, 8.0]))
    assert pairs.o_pair[0] > 0.3
    assert pairs.o_pair[1] < 0.01
    assert pairs.dG.notna().all()


def test_unscored_pairs():
    df = _harmonic([0.0, 0.5, 1.0])
    states = list(df.columns)
    df.iloc[np.flatnonzero(df.index.isin([states[2]])), df.columns.get_loc(states[1])] = np.nan
    pairs = ar.adjacent_pair_overlaps(df)
    assert list(pairs.status) == ["ok", "unscored"]
    no_samples = df[~df.index.isin([states[2]])]
    assert ar.adjacent_pair_overlaps(no_samples).status.iloc[1] == "unscored"


# --------------------------------------------------------------------------------------------------
# Classification on real cycle orders
# --------------------------------------------------------------------------------------------------
def _in_solvent(con=(-8.0, -2.0, 4.0), charges=(0.0, 0.5, 1.0), lj=(0.1, 0.2), screening=(0.5,)):
    return CycleSteps(conformation_forces=list(con), orientational_forces=[c + 4.0 for c in con],
                      charges_windows=list(charges), external_dielectic=[], lj_windows=list(lj),
                      screening_windows=list(screening), in_solvent=True)


def _bands(order):
    return [ar.classify_pair(a, b, ENDSTATE) for a, b in zip(order, order[1:])]


def test_in_solvent_complex_bands():
    cs = _in_solvent()
    assert _bands(cs.complex_order) == [
        "lj", "lj", "lj",                    # no_interactions -> lj_0.1 -> lj_0.2 -> interactions
        "screening", "screening",            # interactions -> screening_0.5 -> electrostatics q0
        "charge", "charge",                  # q 0 -> 0.5 -> 1 at the max restraint
        "restraint", "restraint",            # q1 anchor (4.0) -> -2 -> -8
        "endstate",
    ]
    coords = [ar.band_coordinate(s, "restraint") for s in cs.complex_order[-4:-1]]
    assert coords == [4.0, -2.0, -8.0]


def test_in_solvent_apo_bands():
    cs = _in_solvent()
    assert _bands(cs.ligand_order) == ["endstate", "restraint", "restraint", "charge", "charge"]
    assert [ar.band_coordinate(s, "charge") for s in cs.ligand_order[-3:]] == [1.0, 0.5, 0.0]
    assert _bands(cs.receptor_order) == ["endstate", "restraint", "restraint"]


def test_in_solvent_without_lj_or_screening_windows():
    cs = _in_solvent(lj=(), screening=())
    assert _bands(cs.complex_order)[:2] == ["lj", "screening"]


def test_gas_cycle_bands():
    cs = CycleSteps(conformation_forces=[-8.0, 4.0], orientational_forces=[-4.0, 8.0],
                    charges_windows=[0.0, 1.0], external_dielectic=[2.0, 20.0])
    # no_interactions -> interactions (gas) -> eps 2 -> eps 20 -> water q0 -> q1 -> restraints -> endstate
    assert _bands(cs.complex_order) == ["lj", "dielectric", "dielectric", "dielectric", "charge", "restraint", "endstate"]
    assert _bands(cs.ligand_order) == ["endstate", "restraint", "desolvation", "charge"]
    assert _bands(cs.receptor_order) == ["endstate", "restraint", "dielectric", "dielectric", "dielectric"]
    lam = [ar.band_coordinate(s, "dielectric") for s in cs.complex_order[1:5]]
    assert lam == sorted(lam) and lam[0] == 0.0


# --------------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------------
def _schedule(cs, lj=(0.1, 0.2), screening=(0.5,)):
    return ar.Schedule(restraint=list(zip(cs.conformation_forces, cs.orientational_forces)),
                       charges=list(cs.charges_windows), lj=list(lj), screening=list(screening),
                       dielectric=[], endstate_label=ENDSTATE)


def _table(order, overlap=lambda a, b: 0.3):
    return pd.DataFrame([dict(i=i, a=a, b=b, o_pair=overlap(a, b), status="ok")
                         for i, (a, b) in enumerate(zip(order, order[1:]))])


def _tables(cs, **legs):
    out = {}
    for leg in ("complex", "ligand", "receptor"):
        out[leg] = _table(getattr(cs, f"{leg}_order"), legs.get(leg, lambda a, b: 0.3))
    return out


def _weak_between(band, lo, hi, value=0.001):
    def overlap(a, b):
        if ar.classify_pair(a, b, ENDSTATE) != band:
            return 0.3
        pair = sorted((ar.band_coordinate(a, band), ar.band_coordinate(b, band)))
        return value if pair == [lo, hi] else 0.3
    return overlap


def test_converged_when_nothing_is_weak():
    cs = _in_solvent()
    plan = ar.plan_refinement(_tables(cs), _schedule(cs), 0.03)
    assert plan.reason == "converged" and plan.n_insertions == 0
    assert set(plan.records.status) == {"ok", "endstate_report"}


def test_weak_restraint_gap_in_one_leg_fills_every_leg():
    cs = _in_solvent()
    plan = ar.plan_refinement(_tables(cs, ligand=_weak_between("restraint", -2.0, 4.0)), _schedule(cs), 0.03)
    assert plan.insertions == {"restraint": [(1.0, 5.0)]}
    assert plan.n_md_windows == 3
    filled = plan.records[plan.records.status == "weak_fill"]
    assert set(filled.leg) == {"complex", "ligand", "receptor"}  # the shared gap is marked in every leg


def test_complex_and_ligand_charge_gaps_merge():
    cs = _in_solvent()
    weak = _weak_between("charge", 0.5, 1.0)
    plan = ar.plan_refinement(_tables(cs, complex=weak, ligand=weak), _schedule(cs), 0.03)
    assert plan.insertions == {"charge": [0.75]}
    assert plan.n_md_windows == 2


def test_lj_split_stops_at_the_floor():
    cs = _in_solvent(lj=(0.01,))
    plan = ar.plan_refinement(_tables(cs, complex=_weak_between("lj", 0.0, 0.01)), _schedule(cs, lj=(0.01,)), 0.03)
    assert plan.insertions == {"lj": [0.005]}
    cs = _in_solvent(lj=(0.008,))
    plan = ar.plan_refinement(_tables(cs, complex=_weak_between("lj", 0.0, 0.008)), _schedule(cs, lj=(0.008,)), 0.03)
    assert plan.n_insertions == 0 and plan.reason == "pool-exhausted"
    assert plan.band_reasons["lj"] == "pool-exhausted"


def test_split_too_narrow_for_three_decimals():
    cs = _in_solvent(con=(-8.0, 2.928, 2.929, 4.0))
    weak = _weak_between("restraint", 2.928, 2.929)
    plan = ar.plan_refinement(_tables(cs, complex=weak), _schedule(cs), 0.03, min_steps={"restraint": 0.0})
    assert plan.n_insertions == 0 and plan.reason == "pool-exhausted"


def test_explicit_pool_is_used():
    cs = _in_solvent()
    weak = _weak_between("charge", 0.0, 0.5)
    plan = ar.plan_refinement(_tables(cs, complex=weak), _schedule(cs), 0.03, pools={"charge": [0.1, 0.3, 0.9]})
    assert plan.insertions == {"charge": [0.3]}


def test_endstate_gap_is_reported_not_filled():
    cs = _in_solvent()
    endstate_weak = lambda a, b: 0.001 if ENDSTATE in (a[0], b[0]) else 0.3
    plan = ar.plan_refinement(_tables(cs, complex=endstate_weak), _schedule(cs), 0.03)
    assert plan.n_insertions == 0 and plan.reason == "converged"
    assert len(plan.endstate_gaps) == 1 and plan.endstate_gaps.leg[0] == "complex"


def test_anchors_are_never_produced():
    cs = _in_solvent()
    everything_weak = lambda a, b: 0.0
    plan = ar.plan_refinement(_tables(cs, complex=everything_weak, ligand=everything_weak,
                                      receptor=everything_weak), _schedule(cs), 0.03)
    con = [c for c, _ in plan.insertions["restraint"]]
    assert all(-8.0 < c < 4.0 for c in con)
    assert all(0.0 < q < 1.0 for q in plan.insertions["charge"])
    assert all(0.0 < x < 1.0 for x in plan.insertions["lj"] + plan.insertions["screening"])


def test_budget_fills_weakest_first():
    cs = _in_solvent()

    def overlap(a, b):
        band = ar.classify_pair(a, b, ENDSTATE)
        if band == "charge" and {a[2], b[2]} == {"0.0", "0.5"}:
            return 0.0001
        if band == "restraint" and {a[3].split("_")[0], b[3].split("_")[0]} == {"-8.0", "-2.0"}:
            return 0.01
        return 0.3
    plan = ar.plan_refinement(_tables(cs, complex=overlap), _schedule(cs), 0.03, max_new_windows=3)
    assert plan.insertions == {"charge": [0.25]}
    assert "weak_budget_deferred" in set(plan.records.status)


def test_disabled_band_is_recorded():
    cs = _in_solvent()
    plan = ar.plan_refinement(_tables(cs, complex=_weak_between("lj", 0.1, 0.2)), _schedule(cs), 0.03,
                              bands=("restraint", "charge"))
    assert plan.n_insertions == 0
    assert plan.band_reasons["lj"] == "disabled"


def test_schedule_drift_raises():
    cs = _in_solvent()
    stale = _schedule(_in_solvent(charges=(0.0, 1.0)))
    with pytest.raises(ValueError, match="drifted"):
        ar.plan_refinement(_tables(cs), stale, 0.03)


def test_unscored_pairs_are_not_planned():
    cs = _in_solvent()
    tables = _tables(cs, complex=_weak_between("charge", 0.0, 0.5))
    tables["complex"].loc[tables["complex"].o_pair < 0.03, "status"] = "unscored"
    plan = ar.plan_refinement(tables, _schedule(cs), 0.03)
    assert plan.n_insertions == 0
    assert "unscored" in set(plan.records.status)


def test_max_rounds_reports_gaps_left_to_fill():
    from implicit_solvent_ddm.workflow_phases import _unresolved_gaps
    cs = _in_solvent()
    plan = ar.plan_refinement(_tables(cs, ligand=_weak_between("restraint", -2.0, 4.0)), _schedule(cs), 0.03)
    left = _unresolved_gaps(plan.records, "max-rounds")
    assert {g["leg"] for g in left} == {"complex", "ligand", "receptor"}
    assert {g["status"] for g in left} == {"weak_rounds_exhausted"}
    assert _unresolved_gaps(plan.records, "converged") == []


# --------------------------------------------------------------------------------------------------
# Extending a real config
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def config():
    from implicit_solvent_ddm.config import Config
    path = os.path.join("implicit_solvent_ddm", "tests", "input_files", "config.yaml")
    with open(path) as fh:
        return Config.from_config(yaml.safe_load(fh))


def test_extend_schedule_adds_windows_and_keeps_anchors(config):
    old_order = {leg: getattr(cycle_steps_from_config(config), f"{leg}_order") for leg in ("complex", "ligand")}
    old_args = config.intermediate_args
    new = ar.extend_schedule(config, {"restraint": [(1.0, 5.0)], "charge": [0.25]})
    args = new.intermediate_args
    assert args.exponent_conformational_forces == [-8.0, -2.0, 1.0, 4.0]
    assert args.exponent_orientational_forces == [-4.0, 2.0, 5.0, 8.0]
    assert list(args.conformational_restraints_forces) == pytest.approx(list(np.exp2([-8.0, -2.0, 1.0, 4.0])))
    assert args.charges_lambda_window == [0.0, 0.25, 0.5, 1.0]
    assert args.strongest_conformational_exponent == old_args.strongest_conformational_exponent
    assert args.boresch_Kr == old_args.boresch_Kr
    new_cs = cycle_steps_from_config(new)
    for leg, states in old_order.items():
        assert set(states) <= set(getattr(new_cs, f"{leg}_order"))  # every cached state keeps its label
    # the input config is untouched
    assert old_args.exponent_conformational_forces == [-8.0, -2.0, 4.0]
    assert config.intermediate_args is old_args and new.inputs is not config.inputs


def test_extend_schedule_rejects_collisions_and_moved_anchors(config):
    with pytest.raises(ValueError, match="collides"):
        ar.extend_schedule(config, {"restraint": [(-2.0, 2.0)]})
    with pytest.raises(ValueError, match="anchor"):
        ar.extend_schedule(config, {"restraint": [(5.0, 9.0)]})
    with pytest.raises(ValueError, match="weakest"):
        ar.extend_schedule(config, {"restraint": [(-9.0, -5.0)]})


def test_schedule_from_config_round_trip(config):
    sched = ar.Schedule.from_config(config)
    cs = cycle_steps_from_config(config)
    pairs = {leg: _table(getattr(cs, f"{leg}_order")) for leg in ("complex", "ligand")}
    plan = ar.plan_refinement(pairs, sched, 0.03, bands=ar.BANDS)
    assert plan.reason == "converged"


# --------------------------------------------------------------------------------------------------
# Config validation and deprecations
# --------------------------------------------------------------------------------------------------
def _raw():
    path = os.path.join("implicit_solvent_ddm", "tests", "input_files", "config.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def test_removed_pilot_keys_warn_and_load():
    from implicit_solvent_ddm.config import Config
    raw = _raw()
    raw["workflow"]["intermediate_states_arguments"].update(pilot_ps=50.0, batch_insertion=True,
                                                             max_adaptive_iterations=10)
    with pytest.warns(UserWarning, match="removed ALS pilot"):
        cfg = Config.from_config(raw)
    assert not hasattr(cfg.intermediate_args, "pilot_ps")


def test_adaptive_lambda_needs_exported_files():
    from implicit_solvent_ddm.config import Config
    raw = _raw()
    raw["workflow_jobs"]["adaptive_lambda"] = True
    raw["hardware_parameters"]["export_intermediate_files"] = False
    with pytest.raises(ValueError, match="export_intermediate_files"):
        Config.from_config(raw)


def test_adaptive_lambda_drops_bands_the_cycle_cannot_fill():
    from implicit_solvent_ddm.config import Config
    raw = _raw()
    raw["workflow_jobs"]["adaptive_lambda"] = True
    with pytest.warns(UserWarning, match="screening"):
        cfg = Config.from_config(raw)  # the test config is the gas cycle
    assert cfg.intermediate_args.als_bands == ["restraint", "charge", "lj"]
    raw["workflow"]["intermediate_states_arguments"]["als_bands"] = ["restraint", "bogus"]
    with pytest.raises(ValueError, match="unknown band"):
        Config.from_config(raw)


def test_extend_schedule_fixes_block_anchors(config):
    from implicit_solvent_ddm import block_mbar
    config.intermediate_args.post_analysis_block_size = 2
    old = cycle_steps_from_config(config)
    new = ar.extend_schedule(config, {"restraint": [(1.0, 5.0)]})
    anchors = new.intermediate_args.block_anchor_states
    assert set(anchors) == {"complex", "ligand", "receptor"}
    for leg in anchors:
        old_order = getattr(old, f"{leg}_order")
        new_order = getattr(cycle_steps_from_config(new), f"{leg}_order")
        old_pairs = block_mbar.required_pairs(old_order, 2, block_mbar.band_boundaries(old_order))
        new_pairs = block_mbar.required_pairs(new_order, 2, block_mbar.band_boundaries(new_order), anchors[leg])
        assert old_pairs <= new_pairs
    again = ar.extend_schedule(new, {"charge": [0.25]})
    assert again.intermediate_args.block_anchor_states == anchors  # fixed once, then kept


# --------------------------------------------------------------------------------------------------
# Window audit helpers and DAG wiring
# --------------------------------------------------------------------------------------------------
def test_restraint_blocks_ignore_order_and_spacing(tmp_path):
    from implicit_solvent_ddm.workflow_phases import _restraint_blocks, _topology_body
    a = "&rst iat = 1, 2,\n r1=0, r2 = 3.0,\n/\n&rst iat = 3, 4,\n r1=0, r2=1.5,\n/\n"
    b = "&rst iat = 3, 4,\n r1=0,  r2=1.5,\n/\n\n&rst iat = 1, 2,\n r1=0, r2 = 3.0,\n/\n"
    assert _restraint_blocks(a) == _restraint_blocks(b)
    assert _restraint_blocks(a) != _restraint_blocks(a.replace("3.0", "3.1"))
    p1, p2 = tmp_path / "a.parm7", tmp_path / "b.parm7"
    p1.write_text("%VERSION  VERSION_STAMP = V0001.000  DATE = 11/03/26  10:00:00\n%FLAG TITLE\nx\n")
    p2.write_text("%VERSION  VERSION_STAMP = V0001.000  DATE = 11/04/26  09:00:00\n%FLAG TITLE\nx\n")
    assert _topology_body(str(p1)) == _topology_body(str(p2))


class _FakeJob:
    """Records the job graph ddm_workflow wires, without running anything."""

    def __init__(self, name="root", log=None):
        self.name, self.log = name, log if log is not None else []

    def _add(self, kind, fn):
        child = _FakeJob(getattr(fn, "__name__", type(fn).__name__), self.log)
        self.log.append((kind, self.name, child.name))
        return child

    def addChildJobFn(self, fn, *a, **k):
        return self._add("child", fn)

    def addFollowOnJobFn(self, fn, *a, **k):
        return self._add("followon", fn)

    def addChild(self, job):
        return self._add("child", job)

    def addFollowOn(self, job):
        return self._add("followon", job)

    def rv(self, *a):
        return (self.name, a)


@pytest.mark.parametrize("als", [False, True])
def test_ddm_workflow_wiring(config, als):
    from implicit_solvent_ddm.implicit_ddm_workflow import ddm_workflow
    config.workflow.adaptive_lambda = als
    root = _FakeJob()
    ddm_workflow(root, config)
    edges = [(kind, a, b) for kind, a, b in root.log if b != "initilized_jobs"]
    names = [b for _, _, b in edges]
    if als:
        assert ("followon", "run_endstate_simulations", "als_round") in edges
        assert ("followon", "als_round", "refine_schedule") in edges
        assert "setup_intermediate_simulations" not in names  # phases 3-7 run inside als_round
    else:
        assert names == ["setup_workflow_components", "run_endstate_simulations",
                         "decompose_system_and_generate_restraints", "setup_intermediate_simulations",
                         "run_intermediate_and_post", "_aggregate_post_output",
                         "compute_free_energy_and_consolidate"]


def test_als_round_wires_audit_before_md(config):
    from implicit_solvent_ddm.workflow_phases import als_round

    class _Store:
        def logToMaster(self, msg):
            pass

    root = _FakeJob("als_round")
    root.fileStore = _Store()
    out = als_round(root, 0, config, endstate=None)
    edges = [(a, b) for kind, a, b in root.log if b != "initilized_jobs"]
    assert edges == [
        ("als_round", "decompose_system_and_generate_restraints"),
        ("decompose_system_and_generate_restraints", "setup_intermediate_simulations"),
        ("setup_intermediate_simulations", "audit_existing_windows"),
        ("audit_existing_windows", "run_intermediate_and_post"),
        ("run_intermediate_and_post", "_aggregate_post_output"),
        ("_aggregate_post_output", "compute_free_energy_and_consolidate"),
    ]
    assert set(out) == {"config", "phase7"}


def test_cached_cell_is_read_once_and_empty_means_rescore(tmp_path):
    from implicit_solvent_ddm.runner import IntermidateRunner
    assert IntermidateRunner._read_cached_cell(str(tmp_path)) is None
    path = tmp_path / "simulation_mdout.parquet.gzip"
    pd.DataFrame({"ENERGY": []}).to_parquet(path)
    assert IntermidateRunner._read_cached_cell(str(tmp_path)) is None
    pd.DataFrame({"ENERGY": [1.0, 2.0]}).to_parquet(path)
    assert list(IntermidateRunner._read_cached_cell(str(tmp_path)).ENERGY) == [1.0, 2.0]
