"""
class that will parse in pandas dataframe for mbar analysis 
"""

import os
import re
import pickle

from pymbar import MBAR
import pandas as pd
from toil.job import Job
from implicit_solvent_ddm.get_dirstruct import Dirstruct
from implicit_solvent_ddm.mdout import min_to_dataframe

from implicit_solvent_ddm.restraints import RestraintMaker
from typing import Union, Optional
from alchemlyb.visualisation import plot_mbar_overlap_matrix

WORKDIR = os.getcwd()

AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184


class ConsolidateData(Job):
    """Consolidate all completed simulation data into .h5 files and export to .cache/ directory

    Attributes
    ----------
    complex_adative_run: tuple[DataFrame, DataFrame, MBAR]
        Complex only simulations (DataFrame, DataFrame, MBAR) DataFrames for the free energies
        differences (Deltaf_ij), error estimates in free energy
        difference (dDeltaf_ij), and the pyMBAR object, which can be
        used to get more detail.
    ligand_adaptive_run: tuple[DataFrame, DataFrame, MBAR]
        Ligand only simulations (DataFrame, DataFrame, MBAR) DataFrames for the free energies
        differences (Deltaf_ij), error estimates in free energy
        difference (dDeltaf_ij), and the pyMBAR object, which can be
        used to get more detail.
    receptor_adaptive_run: tuple[DataFrame, DataFrame, MBAR]
        Receptor only simulations (DataFrame, DataFrame, MBAR) DataFrames for the free energies
        differences (Deltaf_ij), error estimates in free energy
        difference (dDeltaf_ij), and the pyMBAR object, which can be
        used to get more detail.
    flat_botton_run: tuple[tuple[DataFrame, DataFrame, MBAR]
        Results from exponential averaging to get the contribution of flat bottom restraints.
    temperature: float
        Specified temperature used for all simulations.
    max_conformation_force: float
        Maximum strength for conformational restraints.
    max_orientational_force: float
        Maximum strength for orientational restraints.
    boresch_df: RestraintMaker
        An instance of RestraintMaker is used to retrieve the analytically computed Boresch contribution value.
    working_path: str
        Path to working directory
    complex_filename: str
        Name of the complex.
    ligand_filename: str
        Name of the ligand/guest molecule.
    receptor_filename: str
        Name of the receptor.
    plot_overlap_matrix: bool
        If true, an overlap-matrix heatmap PDF is written for each leg (complex, receptor, ligand).
        The raw overlap matrices are written as .h5 regardless of this flag.

    Methods
    -------
    _export_overlap_matrices(self)
       Export the MBAR overlap matrix for all three legs (complex, receptor, ligand): raw .h5 always,
       heatmap PDFs when plot_overlap_matrix is set.
    run(self)
        Runner function to consolidate all the output data.
    """

    def __init__(
        self,
        complex_adative_run,
        ligand_adaptive_run,
        receptor_adaptive_run,
        flat_botton_run,
        temperature: float,
        max_conformation_force,
        max_orientational_force,
        boresch_df: RestraintMaker,
        working_path,
        complex_filename,
        ligand_filename,
        receptor_filename,
        endstate_label: str = "endstate",
        plot_overlap_matrix: bool = False,
        memory: Optional[Union[int, str]] = None,
        cores: Optional[Union[int, float, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ):
        Job.__init__(
            self,
            memory="2G",
            cores=2,
            disk="3G",
            accelerators=None,
            preemptible="false",
            unitName=unitName,
            checkpoint=checkpoint,
            displayName=displayName,
        )
        self.temp = temperature
        self.complex_adative_run = complex_adative_run
        self.receptor_adaptive_run = receptor_adaptive_run
        self.ligand_adaptive_run = ligand_adaptive_run
        self.flat_botton_run = flat_botton_run
        self.max_con_force = str(max_conformation_force)
        self.max_orien_force = str(max_orientational_force)
        # Matches the endstate row's state_label, which names the endstate method.
        self.endstate_label = endstate_label
        self.boresch = boresch_df
        self.complex_name = re.sub(r"\..*", "", os.path.basename(complex_filename))
        self.ligand_name = re.sub(r"\..*", "", os.path.basename(ligand_filename))
        self.receptor_name = re.sub(r"\..*", "", os.path.basename(receptor_filename))
        self.working_path = working_path
        self.plot_overlap_matrix = plot_overlap_matrix

    @property
    def mbar_model(self) -> MBAR:
        """return pyMBAR object"""
        return self.complex_adative_run[0][-1]

    @property
    def complex_mbar_formatted_df(self) -> pd.DataFrame:
        """Get all total energies in kcal/mol for the complex"""
        return self.complex_adative_run[1] * self.kcals_per_Kt

    @property
    def complex_fe(self) -> pd.DataFrame:
        """Get the complex free energies differences in kcal/mol"""
        return self.complex_adative_run[0][0] * self.kcals_per_Kt

    @property
    def complex_error(self) -> pd.DataFrame:
        """Get the complex error estimates in free energy (kcal/mol)"""
        return self.complex_adative_run[0][1] * self.kcals_per_Kt

    @property
    def ligand_mbar_formatted_df(self) -> pd.DataFrame:
        """Get all total energies in kcal/mol for the ligand"""
        return self.ligand_adaptive_run[1] * self.kcals_per_Kt

    @property
    def ligand_fe(self) -> pd.DataFrame:
        """Get the ligand free energies differences in kcal/mol"""
        return self.ligand_adaptive_run[0][0] * self.kcals_per_Kt

    @property
    def ligand_error(self) -> pd.DataFrame:
        """Get the ligand error estimates in free energy (kcal/mol)"""
        return self.ligand_adaptive_run[0][1] * self.kcals_per_Kt

    @property
    def receptor_mbar_formatted_df(self) -> pd.DataFrame:
        """Get all total energies in kcal/mol for the receptor"""
        return self.receptor_adaptive_run[1] * self.kcals_per_Kt

    @property
    def receptor_fe(self):
        """Get the receptor free energies differences in kcal/mol"""
        return self.receptor_adaptive_run[0][0] * self.kcals_per_Kt

    @property
    def receptor_error(self) -> pd.DataFrame:
        """Get the receptor error estimates in free energy (kcal/mol)"""
        return self.receptor_adaptive_run[0][1] * self.kcals_per_Kt

    @property
    def flat_bottom_fe(self):
        """Get the flatbottom restraint contribution in kcal/mol."""
        return self.flat_botton_run[0][0] * self.kcals_per_Kt

    @property
    def kcals_per_Kt(self):
        """kcal/mol unit conversion"""
        return ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * self.temp

    @property
    def _get_ligand_deltaG(self):
        """Get the ligand free energy contribution for DeltaG"""
        return self.ligand_fe.loc[
            (self.endstate_label, "78.5", "1.0", "0.0"),
            [("electrostatics", "0.0", "0.0", self.max_con_force)],
        ].values[0]

    @property
    def _get_receptor_deltaG(self):
        """Get the receptor free energy contribution for DeltaG"""
        return self.receptor_fe.loc[
            (self.endstate_label, "78.5", "1.0", "0.0"),
            [("no_gb", "0.0", "1.0", self.max_con_force)],
        ].values[0]

    @property
    def _get_complex_deltaG(self):
        """Get the complex free energy contribution for DeltaG"""
        return self.complex_fe.loc[
            (
                "no_interactions",
                "0.0",
                "0.0",
                f"{self.max_con_force}_{self.max_orien_force}",
            ),
            [(self.endstate_label, "78.5", "1.0", "0.0_0.0")],
        ].values[0]

    @property
    def _flat_bottom_contribution(self):
        """Get the flat bottom restraints free energy contribution for DeltaG"""
        return self.flat_bottom_fe.loc[
            (
                ("no_flat_bottom", "78.5", "1.0", "0.0_0.0"),
                [(self.endstate_label, "78.5", "1.0", "0.0_0.0")],
            )
        ].values[0]

    @property
    def _get_boresch_standard_state(self):
        """Get the Boresch analytical contribution for DeltaG"""
        return self.boresch.boresch_deltaG["DeltaG"].values[0]

    @property
    def compute_binding_deltaG(self) -> float:
        """Compute the total sum of DeltaG"""
        return (
            self._get_complex_deltaG
            + self._get_ligand_deltaG
            + self._get_receptor_deltaG
            + self._get_boresch_standard_state
            + self._flat_bottom_contribution
        )

    def _export_overlap_matrices(self):
        """Export the MBAR overlap matrix for ALL THREE legs (complex, receptor, ligand).

        Previously only the complex overlap was exported, and only as a figure gated behind
        ``plot_overlap_matrix`` — the receptor and ligand overlap matrices were computed by MBAR and
        then discarded, even though they are needed to judge each leg's lambda schedule. Here we save
        the raw overlap matrix as an HDF5 dataframe for every leg (always — small and machine-readable)
        and, when ``plot_overlap_matrix`` is set, a heatmap PDF per leg.

        The matrix is in thermodynamic-cycle order: ``compute_mbar`` reorders the dataframe columns to
        ``matrix_order.CycleSteps.<leg>_order`` and builds MBAR from that, so overlap element (i, j)
        corresponds to states ``order[i], order[j]``. We label the exported dataframe's rows/columns
        with those ordered state tuples (carried by ``run[1].columns``, the returned ``df_mbar``) so the
        ordering is explicit in the file rather than a bare 0..N-1 integer axis.
        """
        output_path = os.path.join(
            f"{self.working_path}", f".cache/{self.complex_name}"
        )
        # name -> the (mbar_result, df_mbar) tuple returned by compute_mbar; [0][-1] is the MBAR object,
        # [1] is df_mbar whose columns are the matrix_order-ordered (state, extdiel, charge, restraints).
        legs = {
            self.complex_name: self.complex_adative_run,
            f"receptor_{self.receptor_name}": self.receptor_adaptive_run,
            f"ligand_{self.ligand_name}": self.ligand_adaptive_run,
        }
        for name, run in legs.items():
            overlap = run[0][-1].compute_overlap()["matrix"]
            # readable, ordered labels from the matrix_order column tuples (same order MBAR used)
            labels = ["|".join(map(str, state)) for state in run[1].columns]
            overlap_df = pd.DataFrame(overlap, index=labels, columns=labels)
            # raw overlap matrix (machine-readable; always exported), rows/cols in matrix_order order
            overlap_df.to_hdf(
                f"{output_path}/{name}_O_MBAR.h5", key="df", mode="w"
            )
            # heatmap figure (only when requested; needs matplotlib)
            if self.plot_overlap_matrix:
                axis = plot_mbar_overlap_matrix(overlap)
                axis.figure.savefig(
                    f"{output_path}/{name}_O_MBAR.pdf",
                    bbox_inches="tight",
                    pad_inches=0.0,
                )
                axis.figure.clear()

    def run(self, fileStore):
        output_path = os.path.join(
            f"{self.working_path}", f".cache/{self.complex_name}"
        )

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # parse out formatted dataframe
        self.complex_mbar_formatted_df.to_hdf(
            f"{output_path}/{self.complex_name}_formatted.h5", key="df", mode="w"
        )
        self.receptor_mbar_formatted_df.to_hdf(
            f"{output_path}/receptor_{self.receptor_name}_formatted.h5",
            key="df",
            mode="w",
        )
        self.ligand_mbar_formatted_df.to_hdf(
            f"{output_path}/ligand_{self.ligand_name}_formatted.h5", key="df", mode="w"
        )

        # parse out free energies differences
        self.complex_fe.to_hdf(
            f"{output_path}/{self.complex_name}_fe.h5", key="df", mode="w"
        )
        self.receptor_fe.to_hdf(
            f"{output_path}/receptor_{self.receptor_name}_fe.h5", key="df", mode="w"
        )
        self.ligand_fe.to_hdf(
            f"{output_path}/ligand_{self.ligand_name}_fe.h5", key="df", mode="w"
        )

        # parse out error estimates in free energy
        self.complex_error.to_hdf(
            f"{output_path}/{self.complex_name}_error.h5", key="df", mode="w"
        )
        self.receptor_error.to_hdf(
            f"{output_path}/receptor_{self.receptor_name}_error.h5", key="df", mode="w"
        )
        self.ligand_error.to_hdf(
            f"{output_path}/ligand_{self.ligand_name}_error.h5", key="df", mode="w"
        )

        # parse out boresch restraints dataframe

        self.boresch.boresch_deltaG.to_hdf(
            f"{output_path}/boresch_{self.complex_name}.h5", key="df", mode="w"
        )
        deltaG_df = pd.DataFrame()

        fileStore.logToMaster(
            f"BORESCH standard state {self._get_boresch_standard_state}\n"
        )

        fileStore.logToMaster(f"Ligand Delta G: {self._get_ligand_deltaG}")
        fileStore.logToMaster(f"Receptor Delta G: {self._get_receptor_deltaG}\n")

        fileStore.logToMaster(
            f"Complex unique index keys:\n {self.complex_fe.index.unique()}\n"
        )

        fileStore.logToMaster(f"Complex Delta G: {self._get_complex_deltaG}\n")

        deltaG_df[f"{self.ligand_name}_endstate->no_charges"] = [
            self._get_ligand_deltaG
        ]
        deltaG_df[f"{self.receptor_name}_endstate->no_gb"] = [self._get_receptor_deltaG]
        deltaG_df["boresch_restraints"] = [self._get_boresch_standard_state]
        deltaG_df["flat_bottom_contribution"] = [self._flat_bottom_contribution]
        deltaG_df[f"{self.complex_name}_no-interactions->endstate"] = [
            self._get_complex_deltaG
        ]
        deltaG_df["deltaG"] = [self.compute_binding_deltaG]

        deltaG_df.to_hdf(
            f"{output_path}/deltaG_{self.complex_name}.h5", key="df", mode="w"
        )
        # pickle out complex pymbar model
        filehandler = open(f"{output_path}/pymbar_object_{self.complex_name}", "wb")
        pickle.dump(self.mbar_model, filehandler)
        filehandler.close()

        # export MBAR overlap matrices for all three legs (raw .h5 always; PDFs when requested)
        self._export_overlap_matrices()


def create_mdout_dataframe(
    job,
    directory_args: dict,
    dirstruct: str,
    output_dir: str,
    compress: bool = True,
    mdout_id=None,
) -> pd.DataFrame:
    sim = Dirstruct("mdgb", directory_args, dirstruct=dirstruct)

    # run_args are parsed from the LOGICAL output_dir path (state labels etc.), which is
    # just string parsing and needs no file on disk. The mdout CONTENT comes either from
    # the network output_dir (default) or, when export is off, from the jobStore via the
    # promised FileID (mdout_id) -- so post-analysis never reads the network.
    # Parse labels from the DIRECTORY path directly. Do NOT append "/mdout": fromPath2Dict
    # only strips a trailing component when os.path.isfile() is true, so with export off (the
    # mdout lives only in the jobStore, never written to output_dir) the "mdout" segment would
    # NOT be stripped -> every field shifts by one -> state_label lands in traj_igb (dropped
    # from the MBAR index) -> distinct windows collapse into duplicate rows. output_dir is the
    # directory, so this is correct whether or not the mdout file exists on disk.
    run_args = sim.dirStruct.fromPath2Dict(output_dir)
    if mdout_id is not None:
        mdout = job.fileStore.readGlobalFile(mdout_id)
    else:
        mdout = f"{output_dir}/mdout"
        job.log(f"List files in postprocess directory {os.listdir(output_dir)}\n")
    data = min_to_dataframe(mdout)

    # data["traj_state_label"] = run_args["traj_state_label"]
    # data["state_label"] = run_args["state_label"]

    data["solute"] = run_args["topology"]
    data["parm_state"] = run_args["state_label"]
    data["traj_state"] = run_args["traj_state_label"]
    data["Frames"] = data.index
    data["charge"] = run_args["charge"]
    data["traj_charge"] = run_args["traj_charge"]
    data["parm_restraints"] = run_args["conformational_restraint"]
    data["traj_restraints"] = run_args["trajectory_restraint_conrest"]
    data["extdiel"] = run_args["extdiel"]
    data["traj_extdiel"] = run_args["traj_extdiel"]
    # complex datastructure
    if "trajectory_restraint_orenrest" in run_args.keys():
        data["parm_restraints"] = (
            f"{run_args['conformational_restraint']}_{run_args['orientational_restraints']}"
        )
        data["traj_restraints"] = (
            f"{run_args['trajectory_restraint_conrest']}_{run_args['trajectory_restraint_orenrest']}"
        )

    if compress:
        data.to_parquet(
            f"{output_dir}/simulation_mdout.parquet.gzip", compression="gzip"
        )
        # data.to_parquet(f"{output_dir}/simulation_mdout.zip",  compression="gzip")

    # if os.path.exists(mdout):
    #     os.remove(mdout)

    return data
