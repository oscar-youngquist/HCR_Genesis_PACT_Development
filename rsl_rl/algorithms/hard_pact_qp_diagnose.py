"""Explicitly opt-in bounded capture. CPU copies/synchronization are diagnostic only."""
from collections import deque, Counter
from dataclasses import asdict, fields
from pathlib import Path
import io
import torch


def owned_cpu(value, memo=None):
    memo = {} if memo is None else memo
    if torch.is_tensor(value):
        if id(value) not in memo:
            memo[id(value)] = value.detach().to("cpu", copy=True)
        return memo[id(value)]
    if isinstance(value, dict):
        return {k: owned_cpu(v,memo) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [owned_cpu(v,memo) for v in value]
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
    def __init__(self, directory, *, limit=32, byte_limit=2048*1024**2,
                 trigger_nm=1000., healthy_limit=2, history_limit=2, identity=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit, self.byte_limit, self.trigger_nm = limit, byte_limit, trigger_nm
        self.healthy_limit, self.healthy_count = healthy_limit, 0
        self.count = self.bytes_written = self.dropped = 0
        self.history = deque(maxlen=history_limit)
        self.identity = identity or {}
        self.counts, self.coverage, self.coverage_bytes = Counter(), Counter(), Counter()

    def summary(self):
        return dict(schema_version=3, captures=self.count, bytes_written=self.bytes_written,
                    dropped_for_budget=self.dropped, counts=dict(self.counts), coverage=dict(self.coverage))

    def before(self, owner, m, data, stage, rows):
        group = owner._diagnostics_phase + "/" + stage
        self.counts[group + "/attempts"] += 1
        self.counts[group + "/rows"] += rows.numel()
        stub = dict(summary_only=True, phase=owner._diagnostics_phase, stage=stage,
                    problem={k:getattr(m,k).detach() for k in ("variable_scale","tau_lower","tau_upper")})
        if self.count >= self.limit or self.bytes_written >= self.byte_limit:
            for backend in owner._backend_instances.values():
                backend.capture_details_enabled = False
            self.dropped += 1
            self.counts[group + "/budget_drops"] += 1
            return stub
        G,h,lo,hi = owner._cupiqp_native_pack(m) if owner._active_solver == "cupiqp" else (m.G,m.h,None,None)
        torque_limit, qmin, qmax, vmax = owner._limits(m.p)
        packet = {"schema_version": 3, "identity": self.identity,
            "config": asdict(owner.cfg), "solver": owner._active_solver,
            "phase": owner._diagnostics_phase, "stage": stage,
            "differentiable": owner._active_differentiable,
            "iteration": getattr(owner, "diagnostic_iteration", lambda: None)(),
            "rows": rows, "data": data,
            "joint_limits": dict(position_lower=qmin,position_upper=qmax,velocity=vmax,
                acceleration=None,  # v3: position/velocity envelope only.
                torque_magnitude=torque_limit,
                torque_rate_lower=data["previous_torque"]-owner.cfg.torque_rate_limit_nm_s*data["dt"].reshape(-1,1),
                torque_rate_upper=data["previous_torque"]+owner.cfg.torque_rate_limit_nm_s*data["dt"].reshape(-1,1),
                names=self.identity.get("joint_names", [f"joint_{j}" for j in range(12)])),
            "problem": {f.name: getattr(m, f.name) for f in fields(m)},
            "tensors": dict(Q=m.Q,p=m.p,G=G,h=h,A=m.A,b=m.b,native_lower=lo,native_upper=hi),
            "unavailable": ["exact solver initialization/preconditioner/factorization state",
                "autograd lease overlap before bounded sequence", "solver state preceding bounded sequence"]}
        # Refuse oversized batches explicitly; never silently save a row subset.
        if tensor_bytes(packet) > self.byte_limit - self.bytes_written:
            self.dropped += 1
            self.counts[group + "/budget_drops"] += 1
            return stub
        return owned_cpu(packet)  # BEFORE backend update/setup may mutate inputs

    def after(self, packet, result=None, accepted=None, error=None):
        if packet is None:
            return
        p = packet["problem"]
        group = packet["phase"] + "/" + packet["stage"]
        reasons = []
        if result is not None:
            raw = result.solution.detach() if packet.get("summary_only") else owned_cpu(result.solution)
            physical = raw * p["variable_scale"]
            violation = torch.maximum(p["tau_lower"]-physical[:,:12], physical[:,:12]-p["tau_upper"]).clamp_min(0).amax(-1)
            finite = torch.isfinite(raw).all(-1)
            snapshot = result.snapshot if packet.get("summary_only") else owned_cpu(result.snapshot)
            status = snapshot.get("status") if snapshot else None
            if status is None:self.counts[group+"/status_unavailable_rows"] += raw.shape[0]
            numerical = (status[:raw.shape[0]].reshape(-1).to(raw.device) == 4) if torch.is_tensor(status) else torch.zeros_like(finite)
            bad = ~finite | (violation > self.trigger_nm) | numerical
            for reason, mask in (("nonfinite",~finite),("extreme_torque",violation>self.trigger_nm),
                                 ("numerical_status",numerical)):
                self.counts[group+"/"+reason+"_rows"] += int(mask.sum())
                if bool(mask.any()): reasons.append(reason)
            if accepted is not None:
                self.counts[group+"/accepted_rows"] += int(accepted.sum())
                self.counts[group+"/rejected_rows"] += int((~accepted).sum())
                if bool((~accepted).any()): reasons.append("rejected")
            if packet.get("summary_only"):
                return
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
            packet["assessment"] = candidate_assessment(packet, raw,
                packet["duality_gap"],packet["duality_gap_rel"])
        else:
            trigger = True
            reasons.append("exception")
            self.counts[group+"/exceptions"] += 1
            if packet.get("summary_only"):
                return
            packet.update(exception=repr(error), failing_rows=torch.arange(packet["tensors"]["p"].shape[0]),
                          accepted=None, raw_primal=None)
        # Acceptance is the exact production certificate + gap policy. Raw
        # candidates remain untouched in this packet; rejection causes can be
        # independently reconstructed from original matrices and profile.
        healthy = not trigger and accepted is not None and bool(accepted.all())
        if healthy: reasons.append("healthy")
        novel = any(self.coverage[group+"/"+r] == 0 for r in reasons)
        # Reserve one quarter per phase/stage; repeated failures cannot crowd
        # out healthy controls or distinct failure reasons within that stratum.
        quota = max(1, (self.limit+3)//4)
        eligible = (novel or self.coverage[group] < max(1,quota//2))
        selected = (trigger or "rejected" in reasons or
                    (healthy and self.coverage[group+"/healthy"] < self.healthy_limit))
        if selected and eligible and self.coverage[group] < quota:
            payload = dict(packet, preceding_updates=[
                {"packet_file":p["_saved_file"]} if "_saved_file" in p else p for p in self.history],
                healthy_reference=healthy)
            stream = io.BytesIO()
            torch.save(payload, stream)
            size = stream.tell()
            budget = self.byte_limit if self.limit < 4 else self.byte_limit//4
            if size <= min(self.byte_limit-self.bytes_written, budget-self.coverage_bytes[group]):
                path = self.directory / f"qp_{self.count:04d}_{packet['phase']}_{packet['stage']}.pt"
                # Exclusive creation prevents accidental replacement of prior evidence.
                with path.open("xb") as target:
                    target.write(stream.getbuffer())
                self.count += 1
                packet["_saved_file"] = path.name
                self.bytes_written += size
                self.healthy_count += int(healthy)
                self.coverage[group] += 1
                self.coverage_bytes[group] += size
                for reason in reasons: self.coverage[group+"/"+reason] += 1
            else:
                self.dropped += 1
                self.counts[group+"/budget_drops"] += 1
        elif selected:
            self.counts[group+"/coverage_skips"] += 1
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
    return {"finite":finite, "primal_feasible":finite & (imax<=tolerance) & (emax<=tolerance),
            "inequality_residual":imax, "equality_residual":emax, "torque_violation_nm":torque}


@torch.no_grad()
def distribution(values):
    """Finite-coordinate distribution; null/NaN is unavailable, not zero."""
    v = values.detach().double().flatten()
    finite = v[torch.isfinite(v)]
    return dict(count=finite.numel(), nonfinite_count=v.numel()-finite.numel(),
        positive_count=(finite>0).sum(),zero_count=(finite==0).sum(),
        mean=finite.mean() if finite.numel() else float("nan"),
        max=finite.max() if finite.numel() else float("nan"),
        percentiles=torch.quantile(finite,finite.new_tensor([.5,.95,.99])) if finite.numel() else None)


@torch.no_grad()
def candidate_assessment(packet, raw, gap=None, relative=None, cfg=None, differentiable=None):
    """Exact production projection/certificate/gap logic, plus original limits.

    Statistics exclude padding by slicing to captured real rows. Recovery joint
    residual rows describe SOFT feasibility; original per-joint limits are
    assessed separately. Status alone is never treated as infeasibility proof.
    """
    from types import SimpleNamespace
    from .hard_pact_qp import HardPACTQPConfig, HardPACTDifferentiableQP, production_gap_pass
    config = dict(packet["config"])
    if packet.get("schema_version",1) < 3:
        config.pop("joint_acceleration_limits_rad_s2",None)
    cfg = cfg or HardPACTQPConfig.from_dict(config)
    diff = packet["differentiable"] if differentiable is None else differentiable
    n = packet["tensors"]["p"].shape[0]
    raw = raw.detach()[:n]
    p = {k:(v.to(raw) if torch.is_tensor(v) else v) for k,v in packet["problem"].items()}
    checker = SimpleNamespace(cfg=cfg, _maximum=HardPACTDifferentiableQP._maximum)
    profile = HardPACTDifferentiableQP._profile(checker,diff)
    tolerance = (HardPACTDifferentiableQP._normalized_tolerance(checker,raw.dtype)
                 if packet["solver"]=="qpth" else profile["feasibility"])
    physical = raw*p["variable_scale"]
    post = physical.clone()
    post[:,:12] = post[:,:12].clamp(p["tau_lower"],p["tau_upper"])
    if "data" in packet:
        stance = packet["data"]["contact_probability"].to(raw)>=cfg.contact_threshold
        # Preserve stage-specific NaN semantics as well as normal arithmetic.
        post[:,12:24] = (post[:,12:24]*stance.repeat_interleave(3,1)
            if packet["stage"]=="recovery" else
            torch.where(stance.repeat_interleave(3,1),post[:,12:24],0.))
    m = SimpleNamespace(**p)
    raw_ok, raw_er, raw_ir = HardPACTDifferentiableQP._certificate(checker,m,raw,tolerance)
    post_ok, er, ir = HardPACTDifferentiableQP._certificate(checker,m,post/p["variable_scale"],tolerance)
    gap = gap[:n].to(raw) if gap is not None else None
    relative = relative[:n].to(raw) if relative is not None else None
    gap_ok = (production_gap_pass(gap,relative,profile,post_ok) if packet["solver"]=="cupiqp"
              else torch.ones_like(post_ok))
    finite = torch.isfinite(raw).all(-1)
    accepted = post_ok & finite & gap_ok
    result = dict(raw_primal_feasible=raw_ok,post_primal_feasible=post_ok,
        production_accepted=accepted,gap_pass=gap_ok,gap_absolute=gap,gap_relative=relative,
        raw_equality=raw_er,raw_inequality=raw_ir,post_equality=er,post_inequality=ir,
        rejection_reasons=dict(nonfinite=~finite,primal=~post_ok,gap=~gap_ok),
        real_rows=n,original_hard_joint_satisfied=None,groups={})
    for label,x in (("raw",physical),("post_projection",post)):
        for units,G,h in (("normalized",p["G"],p["h"]),("physical",p["physical_G"],p["physical_h"])):
            residual = (G@(x/p["variable_scale"] if units=="normalized" else x)[...,None]).squeeze(-1)-h
            groups=[("actuator_absolute" if post.shape[1]==48 else "actuator_rate",slice(0,24)),
                    ("joint_soft" if packet["stage"]=="recovery" else "joint",slice(24,48)),
                    ("friction",slice(48,68))]
            groups += ([("rate_soft",slice(68,92)),("joint_slack",slice(92,104)),("rate_slack",slice(104,116))]
                       if post.shape[1]==48 else [("slack",slice(68,None))])
            for name,sl in groups:
                values=residual[:,sl].clamp_min(0)
                result["groups"][f"{label}/{units}/{name}"] = dict(all=distribution(values),accepted=distribution(values[accepted]))
    limits = packet.get("joint_limits")
    if limits is None:
        result["joint_unavailable"] = "v1 packet did not store original joint limits"
        return result
    d=packet["data"]
    q,v,dt=(d[k].to(raw) for k in ("joint_position","joint_velocity","dt"))
    dt=dt.reshape(-1,1)
    qmin,qmax,vmax=(limits[k].to(raw) for k in ("position_lower","position_upper","velocity"))
    amax = limits.get("acceleration")
    amax = amax.to(raw) if amax is not None else None
    beta=cfg.position_integration_coefficient
    lows=[(-vmax-v)/dt,(qmin-q-dt*v)/(beta*dt.square())]
    highs=[(vmax-v)/dt,(qmax-q-dt*v)/(beta*dt.square())]
    if amax is not None:
        lows.insert(0,-amax.expand_as(q)); highs.insert(0,amax.expand_as(q))
    lower,upper=torch.stack(lows,-1),torch.stack(highs,-1)
    lo,li=lower.max(-1);hi,ui=upper.min(-1)
    a=(p["acceleration_map"]@post[...,None]).squeeze(-1)[:,6:]+p["acceleration_offset"][:,6:]
    vn=v+dt*a;qn=q+dt*v+beta*dt.square()*a
    violations=dict(position_rad=torch.maximum(qmin-qn,qn-qmax).clamp_min(0),
        velocity_rad_s=(vn.abs()-vmax).clamp_min(0))
    if amax is not None:
        violations["acceleration_rad_s2"]=(a.abs()-amax).clamp_min(0)
    rate_violation=torch.maximum(limits["torque_rate_lower"].to(raw)-post[:,:12],
        post[:,:12]-limits["torque_rate_upper"].to(raw)).clamp_min(0)
    result["original_hard_rate_satisfied"] = (rate_violation<=1e-6).all(-1)
    result["joint"] = dict(names=limits["names"],interval_family_order=(
        ["velocity","position"] if amax is None else ["acceleration","velocity","position"]),
        tie_rule="first family",lower_by_family=lower,upper_by_family=upper,lower=lo,upper=hi,
        empty=lo>hi,conflict_rad_s2=(lo-hi).clamp_min(0),lower_family=li,upper_family=ui,
        largest_conflict_joint=(lo-hi).argmax(-1),acceleration=a,q_next=qn,dq_next=vn,
        slack_rad_s2=post[:,24:36] if post.shape[1]>24 else torch.zeros_like(a),
        rate_slack_nm=post[:,36:48] if post.shape[1]==48 else torch.zeros_like(a),
        rate_violation_nm=rate_violation,violations=violations,
        statistics={name:{scope:dict(aggregate=distribution(value[mask]),
                    per_joint=[distribution(value[mask,j]) for j in range(12)])
                    for scope,mask in (("all",torch.ones_like(accepted)),("accepted",accepted))}
                    for name,value in violations.items()})
    result["original_hard_joint_satisfied"] = (torch.isfinite(a).all(-1) &
        (violations["position_rad"]<=1e-5).all(-1) & (violations["velocity_rad_s"]<=1e-4).all(-1) &
        (torch.ones_like(accepted) if amax is None else (violations["acceleration_rad_s2"]<=1e-3).all(-1)))
    return result


def select_replay_rows(packet, limit):
    """Round-robin evidence classes: failures, conflicts, slack, healthy controls."""
    n=packet["tensors"]["p"].shape[0]
    assessment=packet.get("assessment",{})
    joint=assessment.get("joint",{})
    choices=[packet.get("failing_rows",torch.empty(0,dtype=torch.long)).tolist()]
    if "empty" in joint: choices.append(joint["empty"].any(-1).nonzero().flatten().tolist())
    if "slack_rad_s2" in joint:
        slack=joint["slack_rad_s2"].amax(-1)
        choices.append([i for i in slack.argsort(descending=True).tolist() if slack[i]>0])
    accepted=packet.get("accepted")
    choices.append(accepted[:n].nonzero().flatten().tolist() if accepted is not None else [])
    selected=[]
    for rank in range(min(n,limit)):
        for group in choices:
            if rank<len(group) and group[rank] not in selected:
                selected.append(group[rank])
                if len(selected)>=limit:return selected
    return selected if limit else []


@torch.no_grad()
def conditioning_audit(packet, raw, count):
    """Bounded selected-row SVD and approximate active-set KKT, not a proof."""
    reports=[]
    for row in select_replay_rows(packet,count):
        p=packet["problem"]
        Q,G,h=(p[k][row].double().cpu() for k in ("Q","G","h"))
        z=raw[row].detach().double().cpu();linear=p["p"][row].double().cpu()
        if not all(torch.isfinite(v).all() for v in (Q,G,h,z,linear)):
            reports.append(dict(row=row,unavailable="nonfinite data"));continue
        eig=torch.linalg.eigvalsh(Q);res=G@z-h;active=res.abs()<1e-4
        C=G[active]
        multipliers=torch.linalg.lstsq(C.T,-(Q@z+linear)).solution if C.numel() else z.new_empty(0)
        stationarity=Q@z+linear+C.T@multipliers
        reports.append(dict(row=row,row_identity=packet.get("rows",torch.arange(raw.shape[0]))[row],
            q_eigen_min=eig.min(),q_eigen_max=eig.max(),
            q_condition=torch.linalg.cond(Q),active_rank=torch.linalg.matrix_rank(C) if C.numel() else 0,
            active_rows=C.shape[0],approx_stationarity=stationarity.abs().max(),
            negative_multiplier_count=(multipliers<0).sum(),
            approximate_complementarity=(multipliers*res[active]).abs().max() if multipliers.numel() else 0,
            interpretation="Approximate active-set KKT; status or ill-conditioning alone does not prove infeasibility"))
    return reports
