"""Iteration-boundary intervention schedule, independent of objective weights.

Gate evidence uses rolling, complete iteration windows. Tracking is
sample-weighted; failures are events / simulated environment-second. A missing
iteration invalidates its window, not an invitation to reuse reporting totals.
"""
import math
from copy import deepcopy


def legacy_extra_strength_contract(saved, current):
    """Fill absent execution options with legacy DISABLED defaults."""
    if not isinstance(saved,dict) or not isinstance(current,dict):
        return deepcopy(saved)
    result=deepcopy(saved)
    if 'qp_update_mode' in saved:
        for key,value in (('execution_correction_clip_enabled',False),
                          ('execution_correction_hip_limit_nm',12.),
                          ('execution_correction_thigh_limit_nm',12.),
                          ('execution_correction_calf_limit_nm',18.),
                          ('intervention_gate_check_interval',5)):
            if key in current:result.setdefault(key,value)
        for key,value in (('intervention_extra_strength_enabled',False),
                          ('intervention_extra_strength_start',.25)):
            if key in current: result.setdefault(key,value)
    for key in saved.keys() & current.keys():
        if isinstance(saved[key],dict) and isinstance(current[key],dict):
            result[key]=legacy_extra_strength_contract(saved[key],current[key])
    return result


def legacy_intervention_contract(saved, current):
    """Narrow migration for checkpoints predating this schedule.

    Selecting the new mode explicitly opts into its fresh two-solve schedule.
    Only new schedule settings and sampler metadata may change; numerical QP,
    actor, loss, and observation contracts still undergo strict comparison.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return deepcopy(saved)
    result = deepcopy(saved)
    # Missing fields mean DISABLED, never copy an enabled destination setting.
    if 'qp_update_mode' in saved:
        result.setdefault('intervention_extra_strength_enabled', False)
        result.setdefault('intervention_extra_strength_start', .25)
    # v1 had a lifetime-best tracking reference. Only add the new gate settings;
    # retain strict validation of all previously present numerical settings.
    if 'qp_update_mode' in saved and 'intervention_tracking_ema_alpha' not in saved:
        for k in ('intervention_tracking_ema_alpha', 'intervention_tracking_reference_window',
                  'intervention_tracking_reference_quantile', 'intervention_min_tracking'):
            if k in current:
                result[k] = deepcopy(current[k])
    modes = ('every_substep', 'random_one_substep', 'first_plus_random_substep', 'mixed_substeps')
    if 'qp_update_mode' in saved and not any(k.startswith('intervention_') for k in saved):
        for k, v in current.items():
            if k.startswith('intervention_') and not k.startswith('intervention_extra_strength_'):
                result[k] = deepcopy(v)
        if current.get('qp_update_mode') == 'curriculum_substeps' and saved['qp_update_mode'] in modes:
            result['qp_update_mode'] = 'curriculum_substeps'
    if (current.get('mode') == 'curriculum_substeps' and saved.get('mode') in modes
            and saved.get('formulation') == 'masked_torque_force_24'
            and not any(k.startswith('intervention_') for k in saved.get('solver_and_objective_settings', {}))):
        for k in ('mode','problems_per_environment_interval','execution_selection','ppo_anchor_selection'):
            result[k] = deepcopy(current[k])
    for k in saved.keys() & current.keys():
        if isinstance(saved[k], dict) and isinstance(current[k], dict):
            result[k] = legacy_intervention_contract(saved[k], current[k])
    return result


def intervention_probabilities(progress):
    knots = ((1., 0., 0.), (.75, .20, .05), (.50, .30, .20),
             (.25, .25, .50), (0., 0., 1.))
    if not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError('Intervention progress must lie in [0,1]')
    k = min(int(progress * 4), 3)
    t = progress * 4 - k
    return tuple(a + t*(b-a) for a, b in zip(knots[k], knots[k+1]))


class InterventionCurriculum:
    # Numeric reasons are compact TensorBoard scalars, documented in the guide.
    RAMP, SETTLING, EVIDENCE, TRACKING, FAILURES, OPEN, COMPLETE = range(7)

    def __init__(self, cfg, actor_ramp_iterations=0):
        for name in ('intervention_settling_iterations', 'intervention_ramp_iterations',
                     'intervention_window_iterations', 'intervention_tracking_reference_window',
                     'intervention_gate_check_interval'):
            value = getattr(cfg, name)
            if not math.isfinite(value) or int(value) != value or value < (0 if 'settling' in name else 1):
                raise ValueError('Invalid '+name)
        for name in ('intervention_tracking_ratio', 'intervention_failure_ratio',
                     'intervention_failure_allowance'):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid '+name)
        if not 0 < cfg.intervention_tracking_ratio <= 1 or cfg.intervention_failure_ratio < 1:
            raise ValueError('Tracking ratio must be in (0,1]; failure ratio >= 1')
        if not (0 < cfg.intervention_tracking_ema_alpha <= 1
                and 0 <= cfg.intervention_tracking_reference_quantile <= 1
                and math.isfinite(cfg.intervention_min_tracking) and cfg.intervention_min_tracking >= 0):
            raise ValueError('Invalid intervention tracking EMA/quantile/minimum')
        self.tracking_ema_alpha = cfg.intervention_tracking_ema_alpha
        self.tracking_reference_window = cfg.intervention_tracking_reference_window
        self.tracking_reference_quantile = cfg.intervention_tracking_reference_quantile
        self.min_tracking = cfg.intervention_min_tracking
        self.tracking_ema = None
        self.tracking_history = []
        self.tracking_threshold = None
        correction_end = (cfg.warmup_iterations + cfg.correction_ramp_start_offset +
                          cfg.correction_ramp_duration
                          if cfg.correction_ramp_enabled and cfg.correction_ramp_duration > 0
                          else cfg.warmup_iterations)
        self.ramp_end = max(cfg.warmup_iterations, correction_end,
                            cfg.warmup_iterations + actor_ramp_iterations)
        self.start = self.ramp_end + cfg.intervention_settling_iterations
        self.duration = cfg.intervention_ramp_iterations
        self.window = cfg.intervention_window_iterations
        self.check_interval = cfg.intervention_gate_check_interval
        self.evidence = []
        self.last_check = None
        self.check_count = 0
        self.tracking_ratio = cfg.intervention_tracking_ratio
        self.failure_ratio = cfg.intervention_failure_ratio
        self.failure_allowance = cfg.intervention_failure_allowance
        self.progress = 0.
        self.last_iteration = -1
        self.tracking_reference = self.failure_reference = None
        self.tracking_mean = self.failure_rate = None
        self.gate = False
        self.reason = self.EVIDENCE
        self.window_start = None
        self.window_iterations = 0
        self.tracking_sum = self.tracking_count = self.failures = self.exposure = 0.
        self.window_valid = True
        self.snapshot_iteration = None
        self.snapshot_progress = 0.

    def begin(self, iteration):
        if self.snapshot_iteration != iteration:
            self.snapshot_iteration = iteration
            self.snapshot_progress = self.progress
        return intervention_probabilities(self.snapshot_progress)

    def finish(self, iteration, tracking, count, failures, exposure):
        if iteration <= self.last_iteration:
            return
        consecutive = self.last_iteration < 0 or iteration == self.last_iteration + 1
        self.last_iteration = iteration
        if iteration < self.ramp_end:
            self.reason = self.RAMP
            return
        valid = (tracking is not None and count is not None and failures is not None and exposure is not None
                 and all(math.isfinite(v) for v in (tracking, count, failures, exposure))
                 and count > 0 and exposure > 0 and failures >= 0 and consecutive)
        # Use only the decision from a PREVIOUS completed window. Missing
        # evidence immediately pauses future progression until a valid window.
        if valid and self.gate and iteration >= self.start:
            self.progress = min(1., self.progress + 1./self.duration)
        if not valid:
            self.gate = False
            self.reason = self.EVIDENCE
            self.evidence = []
            self.last_check = None
        if self.window_start is None:
            self.window_start = iteration
        self.window_iterations += 1
        self.window_valid &= valid
        if valid:
            self.evidence.append((iteration,float(tracking)*float(count),float(count),float(failures),float(exposure)))
            self.evidence = self.evidence[-self.window:]
            # Same EMA and rolling quantile convention as domain randomization.
            # Keep decisions windowed, but let obsolete easy-domain scores age out.
            alpha = self.tracking_ema_alpha
            self.tracking_ema = (tracking if self.tracking_ema is None else
                                 (1-alpha)*self.tracking_ema + alpha*tracking)
            self.tracking_history.append(self.tracking_ema)
            self.tracking_history = self.tracking_history[-self.tracking_reference_window:]
            self.tracking_sum += tracking * count
            self.tracking_count += count
            self.failures += failures
            self.exposure += exposure
        if iteration - self.window_start + 1 >= self.window:
            complete = (self.window_valid and self.window_iterations == self.window)
            if complete:
                # Best survival reference advances ONLY on disjoint windows.
                rate = self.failures / self.exposure
                self.failure_reference = rate if self.failure_reference is None else min(self.failure_reference,rate)
            self.window_start = None
            self.window_iterations = 0
            self.tracking_sum = self.tracking_count = self.failures = self.exposure = 0.
            self.window_valid = True
        if len(self.evidence) == self.window:
            self.tracking_mean = sum(x[1] for x in self.evidence)/sum(x[2] for x in self.evidence)
            self.failure_rate = sum(x[3] for x in self.evidence)/sum(x[4] for x in self.evidence)
            if self.failure_reference is None:
                # First complete consecutive window establishes the baseline;
                # don't wait for an older invalid block's alignment to expire.
                self.failure_reference = self.failure_rate
                self.window_start = None
                self.window_iterations = 0
                self.tracking_sum = self.tracking_count = self.failures = self.exposure = 0.
                self.window_valid = True
            if self.failure_reference is not None and (self.last_check is None or iteration-self.last_check >= self.check_interval):
                self.last_check = iteration
                self.check_count += 1
                ordered = sorted(self.tracking_history)
                self.tracking_reference = ordered[int(self.tracking_reference_quantile*(len(ordered)-1))]
                self.tracking_threshold = max(self.min_tracking,self.tracking_ratio*self.tracking_reference)
                good_tracking = self.tracking_ema >= self.tracking_threshold
                good_failures = self.failure_rate <= self.failure_ratio*self.failure_reference+self.failure_allowance
                self.gate = good_tracking and good_failures
                self.reason = self.TRACKING if not good_tracking else self.FAILURES if not good_failures else self.OPEN
        if iteration < self.start:
            self.reason = self.SETTLING
        elif self.progress == 1:
            self.reason = self.COMPLETE

    def metrics(self):
        p = intervention_probabilities(self.snapshot_progress)
        unavailable = lambda x: float('nan') if x is None else x
        return dict(progress=self.snapshot_progress, probability_2=p[0], probability_3=p[1],
                    probability_4=p[2], expected_solves=sum(n*v for n,v in zip((2,3,4),p)),
                    gate_open=int(self.gate), gate_reason=self.reason,
                    window_ready=int(len(self.evidence)==self.window), window_iterations=len(self.evidence),
                    gate_checks=self.check_count,
                    tracking_mean=unavailable(self.tracking_mean),
                    tracking_ema=unavailable(self.tracking_ema),
                    tracking_threshold=unavailable(self.tracking_threshold),
                    tracking_reference=unavailable(self.tracking_reference),
                    failures_per_env_second=unavailable(self.failure_rate),
                    failure_reference=unavailable(self.failure_reference))

    def state_dict(self):
        return dict(version=3, **deepcopy(vars(self)))

    def load_state_dict(self, state):
        state = deepcopy(state)
        if state.get('version') == 1:
            # Preserve progress, timing and best failure evidence; lifetime tracking
            # maxima cannot reconstruct a rolling history. Recollect a full window.
            for key in ('tracking_ema_alpha', 'tracking_reference_window',
                        'tracking_reference_quantile', 'min_tracking', 'tracking_ema',
                        'tracking_history', 'tracking_threshold'):
                state[key] = deepcopy(getattr(self, key))
            state.update(version=2, tracking_reference=None, tracking_mean=None,
                         gate=False, reason=self.EVIDENCE, window_start=None,
                         window_iterations=0, tracking_sum=0., tracking_count=0.,
                         failures=0., exposure=0., window_valid=True)
        if state.get('version') == 2:
            state.update(version=3,check_interval=self.check_interval,evidence=[],last_check=None,check_count=0,
                         gate=False,reason=self.EVIDENCE,window_start=None,window_iterations=0,
                         tracking_sum=0.,tracking_count=0.,failures=0.,exposure=0.,window_valid=True)
        if state.get('version') != 3 or set(state) != {'version', *vars(self)}:
            raise ValueError('Incompatible intervention curriculum state')
        intervention_probabilities(state['progress'])
        self.__dict__.update(deepcopy({k:v for k,v in state.items() if k != 'version'}))
