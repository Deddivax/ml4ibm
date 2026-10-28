"""Compare Blade FNB cut-cell classification with QuESo."""

from clean_simulation.scripts.studies import run_legacy


if __name__ == "__main__":
    run_legacy("check_blade_fnb_classifier.py")
