"""Failure-only, bounded QP snapshots; no successful-solve synchronization."""
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
import tempfile
import traceback
import warnings
import sys

import torch


def capture_failure(owner, error, tensors, *, relaxed_contact, elastic):
    count = getattr(owner, "_exception_capture_count", 0)
    if not owner.cfg.exception_capture_enabled or count >= owner.cfg.exception_capture_limit:
        return
    owner._exception_capture_count = count + 1
    # Only failure reporting crosses the device boundary. Keep the complete
    # failing chunk: reducing it could hide a batch-size-dependent failure.
    try:
        packages = {}
        for name in ("torch", "cupiqp", "cupy-cuda12x", "qpth"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        payload = {
            "schema_version": 1, "config": asdict(owner.cfg),
            "solver": owner._active_solver,
            "differentiable": owner._active_differentiable,
            "relaxed_contact": relaxed_contact, "elastic": elastic,
            "exception": repr(error), "traceback": traceback.format_exc(),
            "python": platform.python_version(), "packages": packages,
            # Package metadata alone missed Isaac Sim's shadowed Warp module.
            "runtime_modules": {
                name: {"version": str(getattr(sys.modules[name], "__version__", "unknown")),
                       "file": str(getattr(sys.modules[name], "__file__", "unknown"))}
                for name in ("warp", "cupiqp", "cupy") if name in sys.modules
            },
            "device": str(tensors["Q"].device),
            "hardware": (torch.cuda.get_device_name(tensors["Q"].device)
                         if tensors["Q"].is_cuda else platform.processor()),
            "cuda": torch.version.cuda,
            "tensors": {k: v.detach().cpu() if torch.is_tensor(v) else v
                        for k, v in tensors.items()},
        }
        directory = Path(owner.cfg.exception_capture_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="qp_failure_", suffix=".pt", dir=directory, delete=False) as stream:
            torch.save(payload, stream)
            path = stream.name
        owner.last_exception_capture_path = path
        warnings.warn(f"HardPACT {owner._active_solver} exception: {error!r}\n"
                      f"Reproduction snapshot: {path}\n{payload['traceback']}", stacklevel=2)
    except Exception as capture_error:
        warnings.warn(f"QP exception {error!r}; capture failed: {capture_error!r}", stacklevel=2)


def replay_snapshot(payload, device):
    """Replay exact backend matrices, bypassing simulator and assembly.

    Does not reproduce previous solver cache history; a warm-state-only
    failure may therefore require the original training sequence.
    """
    from .hard_pact_qp import HardPACTQPConfig
    from .hard_pact_qp_backends import create_backend
    from qpth.qp import QPFunction
    if payload["config"].get("proximal_rho", 0) != 0:
        raise ValueError(
            "Snapshot matrices contain the removed HardPACT proximal objective; "
            "recapture with the current configuration. Policy weights remain compatible."
        )
    cfg = HardPACTQPConfig.from_dict(payload["config"])
    tensors = {k: v.to(device) if torch.is_tensor(v) else v
               for k, v in payload["tensors"].items()}
    args = [tensors[k] for k in ("Q", "p", "G", "h", "A", "b")]
    if payload["differentiable"]:
        args[1].requires_grad_(True)
    with torch.set_grad_enabled(payload["differentiable"]):
        if payload["solver"] == "qpth":
            eps = cfg.eps or (cfg.eps_float32 if args[0].dtype == torch.float32 else cfg.eps_float64)
            return QPFunction(eps=eps, maxIter=cfg.max_iter,
                              notImprovedLim=cfg.not_improved_limit,
                              check_Q_spd=cfg.check_q_spd, verbose=-1)(*args)
        return create_backend(payload["solver"], cfg).solve(
            *args, differentiable=payload["differentiable"],
            native_lower=tensors.get("native_lower"),
            native_upper=tensors.get("native_upper"),
        ).solution
