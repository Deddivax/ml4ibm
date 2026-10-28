"""Run the blade displacement convergence study for a watertight STL mesh."""

from clean_simulation.scripts.studies import run_legacy


if __name__ == "__main__":
    run_legacy("run_blade_displacement_convergence.py")
