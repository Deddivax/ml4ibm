"""Run cache-cold blade stiffness-construction timing."""

from clean_simulation.scripts.studies import run_legacy


if __name__ == "__main__":
    run_legacy("run_blade_end_to_end_k_timing.py")
