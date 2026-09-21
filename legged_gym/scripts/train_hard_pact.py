"""HardPACT-only entrypoint; pin cuPIQP's runtime before Isaac Sim starts."""
import os
from pathlib import Path
import runpy
import sys


def prepare_solver_runtime(arguments):
    # Isaac Sim inserts its bundled Warp 1.8.2 ahead of site-packages during
    # AppLauncher startup. cuPIQP requires Warp >=1.12; importing the installed
    # compatible runtime first prevents the bundled extension from shadowing
    # it. This changes process import order, never either installed package.
    resolver = runpy.run_path(str(Path(__file__).with_name("hard_pact_solver_selection.py")))
    solvers = resolver["effective_solvers"](arguments)
    if os.environ.get("SIMULATOR") == "isaaclab" and "cupiqp" in solvers:
        import warp
        from packaging.version import Version
        if Version(warp.__version__) < Version("1.12"):
            raise RuntimeError(
                f"HardPACT cuPIQP requires Warp >=1.12; loaded {warp.__version__} "
                f"from {warp.__file__}. Use the cuPIQP Isaac Lab environment."
            )
        # Isaac Sim 5.1 still uses these public-class aliases in annotations
        # and isinstance checks. Warp 1.17 exposes the same classes at its
        # top level; restore only missing aliases in this HardPACT process.
        for name in ("array", "indexedarray"):
            if not hasattr(warp.types, name):
                setattr(warp.types, name, getattr(warp, name))
        print(f"HardPACT cuPIQP runtime: Warp {warp.__version__} ({warp.__file__})", flush=True)


if __name__ == "__main__":
    prepare_solver_runtime(sys.argv[1:])
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
