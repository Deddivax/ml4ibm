"""Evaluate blade convergence on fixed physical interior samples."""

from clean_simulation.scripts.studies import run_legacy


if __name__ == "__main__":
    run_legacy("evaluate_blade_displacement_physical_convergence.py")
