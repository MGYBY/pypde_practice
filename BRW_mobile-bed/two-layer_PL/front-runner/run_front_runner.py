#!/usr/bin/env python3
"""Run from any working directory; one model configuration per process."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, default=root/'case_parameters.ini')
    p.add_argument('--output', type=Path, help='Override output folder (must be empty or new).')
    p.add_argument('--dry-run', action='store_true', help='Write and check the IC without importing the native PyPDE solver.')
    p.add_argument('--uncached', action='store_true', help='Use the public API of the audited backend on each segment (slower).')
    args = p.parse_args()
    os.environ['TLPL_CASE'] = str(args.case.resolve())
    sys.path.insert(0, str(root/'pypde'))
    import two_layer_powerlaw_kp_pypde as model
    from front_runner_runtime import run_case
    run_case(model, output_directory=args.output, dry_run=args.dry_run, force_uncached=args.uncached)


if __name__ == '__main__':
    main()
