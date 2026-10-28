"""Portable training entry point.

This is the supported command for local and remote machines. Runtime
configuration is supplied through the same CUT_CELL_* environment variables.
"""

from run_d20_f4096_aws import main


if __name__ == "__main__":
    main()