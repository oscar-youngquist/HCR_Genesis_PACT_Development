"""Read-only geometric diagnostics; never used as reference or reward masks."""
import torch
from .bounds import geometry

CATEGORIES = ('offset', 'workspace', 'box_ground', 'offset_speed', 'base_speed', 'yaw_rate')

def checked(reference, state, time, inputs):
    """Same accepted-state tolerances as the guard; not articulated IK/collision certification."""
    q, nominal, velocity = reference._relative(state, time, inputs)
    cfg = reference.cfg
    distances = geometry(q, nominal, cfg)[0]
    flags = torch.stack((distances[..., 6] < -1e-4,
        (distances[..., :6] < -1e-4).any(-1), (distances[..., 7:] < -1e-4).any(-1),
        (velocity-inputs[5]).norm(dim=-1) > cfg.compliant_offset_speed_limit+1e-4,
        state[..., 2:4].norm(dim=-1) > cfg.base_speed_limit+1e-4,
        state[..., 5].abs() > cfg.yaw_rate_limit+1e-4), -1)
    offset = (q-nominal).norm(dim=-1)
    return flags, offset, (offset-cfg.max_offset).clamp_min(0), nominal


class SeedDiagnostics:
    """Fixed ring metadata, retained until expiration/reset. No random draws or host reads."""
    def __init__(self, state):
        shape = state.shape[:2]
        self.data = {key: state.new_zeros(shape) for key in (
            'initial_offset_m', 'initial_excess_m', 'elapsed_s', 'recovery_time_s', 'age', 'recovery_age',
            'offset_reduction_m', 'excess_reduction_m', 'nominal_motion_m', 'offset_m', 'excess_m')}
        for key in ('initial_feasible', 'recovered', 'reentered', 'new_violation', 'ever_loaded'):
            self.data[key] = torch.zeros(shape, device=state.device, dtype=torch.bool)
        self.data['initial_flags'] = torch.zeros((*shape, len(CATEGORIES)), device=state.device, dtype=torch.bool)
        self.data['end_flags'] = torch.zeros_like(self.data['initial_flags'])
        self.data['any_flags'] = torch.zeros_like(self.data['initial_flags'])
        self.nominal = state.new_zeros((*shape, 3))
        # Lifecycle counts are cumulative; runner differences them once per rollout.
        # Long training runs exceed float32's exact integer range. Only these
        # tiny cumulative counters need float64; trajectory fields retain state dtype.
        self.censored = torch.zeros(2, device=state.device, dtype=torch.float64)  # expired, reset
        self.exposure = torch.zeros(5, device=state.device, dtype=torch.float64)  # seconds, loaded seconds, |EE F|, |base F|, |moment| integrals

    def retire(self, index, active, reason):
        self.censored[reason] += (active & ~self.data['initial_feasible'][index]
                                  & ~self.data['recovered'][index]).sum()

    def seed(self, column, values):
        flags, offset, excess, nominal = values
        for value in self.data.values():
            value[:, column] = 0
        self.data['initial_flags'][:, column] = flags
        self.data['initial_feasible'][:, column] = ~flags.any(-1)
        self.data['initial_offset_m'][:, column] = offset
        self.data['initial_excess_m'][:, column] = excess
        self.nominal[:, column] = nominal

    def update(self, active, dt, values, any_flags, loaded):
        flags, offset, excess, nominal = values
        d = self.data
        d['elapsed_s'].add_(active*dt)
        d['age'].add_(active)
        feasible = ~flags.any(-1)
        first = active & ~d['initial_feasible'] & ~d['recovered'] & feasible
        d['recovery_time_s'] = torch.where(first, d['elapsed_s'], d['recovery_time_s'])
        d['recovery_age'] = torch.where(first, d['age'], d['recovery_age'])
        # A violation within an interval after an earlier recovery counts as re-entry,
        # even if its endpoint recovers again. First recovery is observed at control boundaries.
        d['reentered'] |= active & d['recovered'] & any_flags.any(-1)
        d['recovered'] |= first
        d['new_violation'] |= active & d['initial_feasible'] & any_flags.any(-1)
        d['ever_loaded'] |= active & loaded[:, None]
        d['offset_reduction_m'] = d['initial_offset_m']-offset
        d['excess_reduction_m'] = d['initial_excess_m']-excess
        # Progress uses the CURRENT nominal; expose command motion rather than
        # misinterpreting a moving nominal as force-induced reference recovery.
        d['nominal_motion_m'] = (nominal-self.nominal).norm(dim=-1)
        d['end_flags'], d['any_flags'] = flags.clone(), any_flags.clone()
        d['offset_m'], d['excess_m'] = offset, excess


class RecoveryRollout:
    """Device reductions over eligible trajectory/horizon observations, not robot means."""
    def __init__(self):
        self.totals = None
        self.maxima = None
        self.groups = ('all', 'initially_feasible', 'initially_infeasible', 'zero_load', 'loaded')
        self.magnitudes = ('initial_offset_m', 'initial_excess_m', 'offset_m', 'excess_m',
                           'offset_reduction_m', 'excess_reduction_m', 'nominal_motion_m')

    def add(self, packet):
        if 'diagnostic_initial_feasible' not in packet:
            return
        d = {k.removeprefix('diagnostic_'): v for k, v in packet.items() if k.startswith('diagnostic_')}
        valid = packet['valid']
        masks = torch.stack((valid, valid & d['initial_feasible'], valid & ~d['initial_feasible'],
                             valid & ~d['ever_loaded'], valid & d['ever_loaded']), -1)
        recovered = ~d['initial_feasible'] & d['recovered']
        fields = dict(count=torch.ones_like(valid), initially_feasible_count=d['initial_feasible'],
            initially_infeasible_count=~d['initial_feasible'], recovered_count=recovered,
            reentry_count=d['reentered'], new_violation_count=d['new_violation'],
            unrecovered_count=~d['initial_feasible'] & ~d['recovered'],
            end_incompatible_count=d['end_flags'].any(-1), any_incompatible_count=d['any_flags'].any(-1),
            recovery_seconds_sum=torch.where(recovered, d['recovery_time_s'], 0),
            recovery_age_sum=torch.where(recovered, d['recovery_age'], 0))
        for i, category in enumerate(CATEGORIES):
            for timing in ('end', 'any', 'initial'):
                fields[timing+'_'+category+'_count'] = d[timing+'_flags'][..., i]
        for name in self.magnitudes:
            fields[name+'_sum'] = d[name]
        self.names = tuple(fields)
        values = torch.stack(tuple(fields.values()), -1).float()
        # One batched reduction for all groups/metrics, not hundreds of small
        # per-metric GPU reductions. Unused slots never enter sufficient statistics.
        total = torch.einsum('nhg,nhm->hgm', masks.float(), values).double()
        magnitude = torch.stack(tuple(d[k] for k in self.magnitudes), -1)
        maximum = torch.where(masks[..., None], magnitude[:, :, None], -torch.inf).amax(0)
        self.totals = total if self.totals is None else self.totals+total
        self.maxima = maximum if self.maxima is None else torch.maximum(self.maxima, maximum)

    def report(self, horizons):
        # Host conversion occurs only after collection. Null means unavailable,
        # including recovery time when no initially-infeasible seed recovered.
        report = {}
        if self.totals is None:
            return report
        totals, maxima = self.totals.tolist(), self.maxima.tolist()
        for j, age in enumerate(horizons):
            groups = {}
            for g, group in enumerate(self.groups):
                raw = dict(zip(self.names, totals[j][g]))
                count = raw.get('count', 0)
                row = dict(raw)
                for name, value in raw.items():
                    if name.endswith('_count'):
                        row[name.removesuffix('_count')+'_fraction'] = value/count if count else None
                    elif name.endswith('_sum') and name not in ('recovery_seconds_sum', 'recovery_age_sum'):
                        row[name.removesuffix('_sum')+'_mean'] = value/count if count else None
                recovered = raw.get('recovered_count', 0)
                row['recovery_time_s_mean'] = raw.get('recovery_seconds_sum', 0)/recovered if recovered else None
                row['recovery_age_mean'] = raw.get('recovery_age_sum', 0)/recovered if recovered else None
                for name, value in zip(self.magnitudes, maxima[j][g]):
                    row[name+'_max'] = value if count else None
                groups[group] = row
            report[str(age)] = groups
        return report
