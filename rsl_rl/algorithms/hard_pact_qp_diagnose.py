"""Explicitly opt-in bounded capture. CPU copies/synchronization are diagnostic only."""
from collections import deque
from dataclasses import asdict, fields
from pathlib import Path
import io
import torch


def owned_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: owned_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [owned_cpu(v) for v in value]
    return value


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(map(tensor_bytes, value))
    return 0


class QPCapture:
    def __init__(self, directory, *, limit=8, byte_limit=512*1024**2,
                 trigger_nm=1000., healthy_limit=2, history_limit=2, identity=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit, self.byte_limit, self.trigger_nm = limit, byte_limit, trigger_nm
        self.healthy_limit, self.healthy_count = healthy_limit, 0
        self.count = self.bytes_written = self.dropped = 0
        self.history = deque(maxlen=history_limit)
        self.identity = identity or {}

    def before(self, owner, m, data, stage, rows):
        if self.count >= self.limit or self.bytes_written >= self.byte_limit:
            for backend in owner._backend_instances.values():
                backend.capture_details_enabled = False
            return None
        G,h,lo,hi = owner._cupiqp_native_pack(m) if owner._active_solver == "cupiqp" else (m.G,m.h,None,None)
        packet = {"schema_version": 1, "identity": self.identity,
            "config": asdict(owner.cfg), "solver": owner._active_solver,
            "phase": owner._diagnostics_phase, "stage": stage,
            "differentiable": owner._active_differentiable,
            "iteration": getattr(owner, "diagnostic_iteration", lambda: None)(),
            "rows": rows, "data": data,
            "problem": {f.name: getattr(m, f.name) for f in fields(m)},
            "tensors": dict(Q=m.Q,p=m.p,G=G,h=h,A=m.A,b=m.b,native_lower=lo,native_upper=hi),
            "unavailable": ["exact solver initialization/preconditioner/factorization state",
                "autograd lease overlap before bounded sequence", "solver state preceding bounded sequence"]}
        # Refuse oversized batches explicitly; never silently save a row subset.
        if tensor_bytes(packet) > self.byte_limit - self.bytes_written:
            self.dropped += 1
            return None
        return owned_cpu(packet)  # BEFORE backend update/setup may mutate inputs

    def after(self, packet, result=None, accepted=None, error=None):
        if packet is None:
            return
        p = packet["problem"]
        if result is not None:
            raw = owned_cpu(result.solution)
            physical = raw * p["variable_scale"]
            violation = torch.maximum(p["tau_lower"]-physical[:,:12], physical[:,:12]-p["tau_upper"]).clamp_min(0).amax(-1)
            finite = torch.isfinite(raw).all(-1)
            snapshot = owned_cpu(result.snapshot)
            status = snapshot.get("status") if snapshot else None
            numerical = (status[:raw.shape[0]].reshape(-1) == 4) if torch.is_tensor(status) else torch.zeros_like(finite)
            bad = ~finite | (violation > self.trigger_nm) | numerical
            packet.update(raw_primal=raw, raw_torque_violation_nm=violation,
                nonfinite=~finite, numerical_failure=numerical,
                duality_gap=owned_cpu(result.duality_gap), duality_gap_rel=owned_cpu(result.duality_gap_rel),
                backend_snapshot=snapshot, accepted=owned_cpu(accepted), failing_rows=bad.nonzero().flatten())
            # Preserve post-projection production primal checks separately from
            # independent RAW checks: a large rejected iterate is not executed.
            candidate = physical.clone()
            candidate[:,:12] = candidate[:,:12].clamp(p["tau_lower"],p["tau_upper"])
            stance = packet["data"]["contact_probability"] >= packet["config"]["contact_threshold"]
            candidate[:,12:24] = torch.where(stance.repeat_interleave(3,1),candidate[:,12:24],0.)
            checks = independent_checks(packet,candidate/p["variable_scale"])
            prefix = "ppo" if packet["differentiable"] else "rollout"
            tol = packet["config"][prefix+"_feasibility_tolerance"]
            packet["rejection_reasons"] = dict(nonfinite=~finite,
                normalized_equality=checks["equality_residual"]>tol,
                normalized_inequality=checks["inequality_residual"]>tol,
                acceptance_policy=~owned_cpu(accepted) if accepted is not None else None)
            if packet["config"][prefix+"_duality_gap_policy"] == "require":
                gap, rel = packet["duality_gap"], packet["duality_gap_rel"]
                packet["rejection_reasons"]["gap_policy"] = (
                    ~(torch.isfinite(gap) & torch.isfinite(rel) &
                      ((gap<=packet["config"][prefix+"_duality_gap_abs"]) |
                       (rel<=packet["config"][prefix+"_duality_gap_rel"])))
                    if gap is not None and rel is not None else "gap unavailable")
            trigger = bool(bad.any())
        else:
            trigger = True
            packet.update(exception=repr(error), failing_rows=torch.arange(packet["tensors"]["p"].shape[0]),
                          accepted=None, raw_primal=None)
        # Acceptance is the exact production certificate + gap policy. Raw
        # candidates remain untouched in this packet; rejection causes can be
        # independently reconstructed from original matrices and profile.
        healthy = not trigger and accepted is not None and bool(accepted.all())
        if trigger or (healthy and self.healthy_count < self.healthy_limit):
            payload = dict(packet, preceding_updates=list(self.history), healthy_reference=healthy)
            stream = io.BytesIO()
            torch.save(payload, stream)
            size = stream.tell()
            if size <= self.byte_limit-self.bytes_written:
                path = self.directory / f"qp_{self.count:04d}_{packet['phase']}_{packet['stage']}.pt"
                # Exclusive creation prevents accidental replacement of prior evidence.
                with path.open("xb") as target:
                    target.write(stream.getbuffer())
                self.count += 1
                self.bytes_written += size
                self.healthy_count += int(healthy)
            else:
                self.dropped += 1
        self.history.append(packet)
        while self.history and tensor_bytes(list(self.history)) > self.byte_limit // 4:
            self.history.popleft()


@torch.no_grad()
def independent_checks(packet, z, tolerance=1e-3):
    """Fixed canonical row-scaled primal checks, independent of tested settings.

    Includes native torque bounds via the original full canonical inequalities.
    This is a primal certificate only, NOT a claim of optimal ground truth.
    """
    z = z.detach().to(torch.float64)
    p = packet["problem"]
    G,h,A,b = (p[k].to(z) for k in ("G","h","A","b"))
    iq = (G@z[...,None]).squeeze(-1)-h
    eq = (A@z[...,None]).squeeze(-1)-b
    imax = iq.clamp_min(0).amax(-1)
    emax = eq.abs().amax(-1) if eq.shape[1] else imax.new_zeros(imax.shape)
    finite = torch.isfinite(z).all(-1)
    x = z*p["variable_scale"].to(z)
    torque = torch.maximum(p["tau_lower"].to(z)-x[:,:12], x[:,:12]-p["tau_upper"].to(z)).clamp_min(0).amax(-1)
    return {"finite":finite, "certified":finite & (imax<=tolerance) & (emax<=tolerance),
            "inequality_residual":imax, "equality_residual":emax, "torque_violation_nm":torque}
