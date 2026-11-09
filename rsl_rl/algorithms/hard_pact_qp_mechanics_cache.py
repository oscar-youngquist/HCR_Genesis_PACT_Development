"""Owned, nominal-only mechanics at the preselected rollout replay anchor."""
import torch

MECHANICS_SHAPES = dict(mass_matrix=(18,18), bias=(18,),
    foot_jacobians=(4,3,18), base_jacobian=(6,18), foot_acceleration_bias=(4,3))
PREFIX = 'sampled_qp_mechanics_'
VERSION = 1  # Canonical deployment mechanics, empty randomized parameters.


def gather_mechanics(fields, global_rows):
    """Gather selected flattened (time, environment) rows, including validity."""
    return {name:value.flatten(0,1)[global_rows].detach()
            for name,value in (fields or {}).items() if name.startswith(PREFIX)}


def allocate_mechanics(rows, reference):
    result = {PREFIX+k: reference.new_zeros((rows,)+shape)
              for k,shape in MECHANICS_SHAPES.items()}
    result[PREFIX+'valid'] = torch.zeros(rows,1,device=reference.device,dtype=torch.bool)
    result[PREFIX+'version'] = torch.full((rows,1),VERSION,device=reference.device,dtype=torch.int16)
    return result


@torch.no_grad()
def copy_mechanics(destination, context, rows, local):
    """Advanced-index assignment copies; never retain mutable context aliases."""
    valid = torch.ones(local.numel(),device=rows.device,dtype=torch.bool)
    for key in MECHANICS_SHAPES:
        value = getattr(context,key).detach()[local]
        target = destination[PREFIX+key]
        if value.dtype != target.dtype or value.device != target.device or value.shape[1:] != target.shape[1:]:
            # Never silently round mechanics into a lower-precision cache.
            valid.zero_()
            continue
        destination[PREFIX+key][rows] = value
        valid &= torch.isfinite(value).flatten(1).all(-1)
    destination[PREFIX+'valid'][rows,0] = valid


def read_mechanics(fields, reference):
    """Return compatible detached tensors/row validity, otherwise recompute all."""
    if fields is None:
        return None
    for key,shape in MECHANICS_SHAPES.items():
        value = fields.get(PREFIX+key)
        if (value is None or value.shape != (reference.shape[0],)+shape
                or value.device != reference.device or value.dtype != reference.dtype):
            return None
    valid,version = fields.get(PREFIX+'valid'),fields.get(PREFIX+'version')
    if valid is None or version is None or valid.shape != (reference.shape[0],1) or version.shape != valid.shape:
        return None
    if valid.device != reference.device or version.device != reference.device:
        return None
    return ({k:fields[PREFIX+k].detach() for k in MECHANICS_SHAPES},
            valid.detach().reshape(-1).bool() & (version.detach().reshape(-1)==VERSION))
