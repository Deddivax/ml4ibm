"""Run hanging-sphere convergence with QuESo interior-cell stiffness."""

from clean_simulation.scripts.studies import run_legacy


if __name__ == "__main__":
    run_legacy("run_queso_inner_cell_convergence_aws.py")
