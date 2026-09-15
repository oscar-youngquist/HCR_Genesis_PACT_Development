"""Collect common-history PACT/DACT–ABL3 reconstruction data (evaluation only).

See reconstruction_eval/README.md for metadata, collection and analysis commands.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reconstruction_eval.collect import main

if __name__ == '__main__':
    main()
