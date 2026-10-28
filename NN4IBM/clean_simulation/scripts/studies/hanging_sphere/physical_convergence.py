"""Evaluate hanging-sphere physical convergence from archived checkpoints."""

from clean_simulation.scripts.studies import run_legacy


if __name__ == "__main__":
    run_legacy("evaluate_remote_displacement_convergence.py")
