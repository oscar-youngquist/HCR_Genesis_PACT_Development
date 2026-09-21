"""Scoped diagnostic observers only: no optimizer changes or extra backward."""
from contextlib import contextmanager
import time
from unittest.mock import patch
import torch

from .hard_pact_qp_backends import CUDAEventProfile, CuPIQPFunction
from .hard_pact_qp_diagnose import distribution
from .hard_pact_qp import _ScaleClipRows


@contextmanager
def diagnostic_runtime(runner, directory, *, gradients):
    from scripts.eval_hard_pact_frozen import write_json
    from . import ppo_hard_pact as ppo
    qp=runner.alg.hard_pact_qp
    timer=CUDAEventProfile(True)
    records=[]; grad_totals={}; samples={}; exceptions=[]; handles=[]; loss_values={}
    def observe(name,g):
        v=g.detach().flatten(); finite=torch.isfinite(v); safe=v.where(finite,0).abs()
        stats=torch.stack((v.new_tensor(v.numel()),(~finite).sum(),safe.sum(),safe.square().sum(),(safe==0).sum()))
        grad_totals[name]=grad_totals.get(name,0)+stats
        if name not in samples:samples[name]=safe[:256].clone()
        # Return None: hooks must not replace the incoming gradient.
    def attach(name,value):
        if gradients and torch.is_tensor(value) and value.requires_grad:
            value.register_hook(lambda g:observe(name,g))
    if gradients:
        for prefix,module in (("actor_critic",runner.alg.actor_critic),("privileged_decoder",runner.alg.decoder)):
            for name,p in module.named_parameters():
                if p.requires_grad:
                    handles.append(p.register_hook(lambda g,key=prefix+"/"+name:observe(key,g)))
    original_solve=qp.solve
    def solve(**kw):
        for key in ("tau_nom","force_pred_world","wrench_pred_world","contact_probability"):
            attach("qp_input/"+key,kw.get(key))
        return original_solve(**kw)
    original_backend=qp._backend_solve
    def forward(m):
        stage="primary" if m.p.shape[-1]==24 else "recovery"
        with timer.measure(qp._diagnostics_phase+"/"+stage+"/forward",m.p):
            return original_backend(m)
    original_backward=CuPIQPFunction.backward
    original_conditioner=_ScaleClipRows.backward
    def conditioned_backward(ctx,g):
        if gradients:observe("qp_input_pre_conditioner/"+ctx.name,g)
        return original_conditioner(ctx,g)
    def backward(ctx,*args):
        stage="primary" if ctx.inputs[1].shape[-1]==24 else "recovery"
        try:
            with timer.measure("ppo/"+stage+"/backward",ctx.inputs[1]):
                return original_backward(ctx,*args)
        except Exception as error:
            exceptions.append(dict(stage=stage,exception=repr(error),iteration=runner.current_learning_iteration))
            raise
    original_iteration=runner._set_hard_pact_qp_iteration
    def iteration(i):
        original_iteration(i)
        # Reuse existing runner collection/update events without enabling any
        # physical/KKT diagnostic work or changing solver numerical profiles.
        for profile in qp.profiles.values():profile.enabled=True
    def loss_wrapper(fn,name):
        def wrapped(*a,**kw):
            value=fn(*a,**kw)
            loss=value[0] if isinstance(value,tuple) else value
            if gradients:
                loss_values[name]=loss_values.get(name,0)+loss.detach()
                attach("qp_loss/"+name,loss)
            return value
        return wrapped
    class Runtime:
        def run_iteration(self,period):
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter(); epoch=runner.current_learning_iteration
            try:runner.learn(1,init_at_random_ep_len=False)
            finally:
                times=timer.finalize()
                for phase,profile in qp.profiles.items():
                    times.update({phase+"/"+k:v for k,v in profile.finalize().items()})
                records.append(dict(iteration=epoch,period=period,cuda_event_ms=times,
                    wall_seconds_including_checkpoint_io=time.perf_counter()-start,
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None))
    try:
        with patch.object(qp,"solve",solve),patch.object(qp,"_backend_solve",forward), \
             patch.object(CuPIQPFunction,"backward",staticmethod(backward)), \
             patch.object(_ScaleClipRows,"backward",staticmethod(conditioned_backward)), \
             patch.object(runner,"_set_hard_pact_qp_iteration",iteration), \
             patch.object(ppo,"projection_loss",loss_wrapper(ppo.projection_loss,"primary")), \
             patch.object(ppo,"recovery_projection_loss",loss_wrapper(ppo.recovery_projection_loss,"recovery")):
            yield Runtime()
    finally:
        for handle in handles:handle.remove()
        write_json(directory/"runtime.json",dict(schema_version=2,iterations=records,
            backward_exceptions=exceptions,gradients_enabled=gradients,
            gradient_totals={k:dict(count=v[0],nonfinite=v[1],abs_sum=v[2],squared_sum=v[3],zero_count=v[4],
                first_256_magnitude_distribution=distribution(samples[k])) for k,v in grad_totals.items()},
            qp_loss_sums=loss_values,unavailable=["optimizer/objective attribution of parameter VJPs",
                "non-Torch/CuPy/Isaac allocator memory", "exact counterfactual capture overhead"],
            semantics="Hooks observe incoming autograd VJPs before clipping/PCGrad; parameter VJPs may include other objectives. No extra backward."))
