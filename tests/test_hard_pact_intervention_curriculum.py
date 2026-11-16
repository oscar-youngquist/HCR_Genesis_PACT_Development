"""CPU-only intervention scheduling, replay sampling and real control fixture."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from rsl_rl.algorithms.hard_pact_intervention_curriculum import (
    InterventionCurriculum, intervention_probabilities,
)
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, qp_interval_selection, qp_substep_mask
from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum
from test_hard_pact_qp_modes import fixture


def config(**kw):
    defaults = dict(qp_update_mode='curriculum_substeps', warmup_iterations=0,
                    correction_ramp_enabled=False, intervention_settling_iterations=0,
                    intervention_ramp_iterations=10, intervention_window_iterations=2)
    return replace(HardPACTQPConfig(), **(defaults | kw))


def finish(s, i, score=.8, count=10, failures=1, exposure=10):
    s.begin(i)
    s.finish(i, score, count, failures, exposure)


@pytest.mark.parametrize('p,expected', [
    (0,(1,0,0)), (.25,(.75,.20,.05)), (.5,(.5,.3,.2)),
    (.75,(.25,.25,.5)), (1,(0,0,1)), (.125,(.875,.1,.025)),
])
def test_interpolation(p, expected):
    assert intervention_probabilities(p) == pytest.approx(expected)


def test_actual_ramps_settling_and_subsequent_iteration_snapshot():
    s = InterventionCurriculum(config(warmup_iterations=10, correction_ramp_enabled=True,
        correction_ramp_start_offset=3, correction_ramp_duration=5,
        intervention_settling_iterations=3), actor_ramp_iterations=12)
    assert (s.ramp_end, s.start) == (22,25)
    for i in range(25):
        finish(s,i)
        assert s.progress == 0
    assert s.gate
    snapshot = s.begin(25)
    s.finish(25,.8,10,1,10)
    assert s.progress == pytest.approx(.1)
    assert s.begin(25) == snapshot  # frozen through the entire update
    assert s.begin(26) != snapshot
    state = deepcopy(s.state_dict())
    s.finish(25,.8,10,1,10)
    assert s.state_dict() == state


def test_complete_baseline_even_without_settling_and_monotone_references():
    s = InterventionCurriculum(config())
    finish(s,0)
    assert not s.gate and s.progress == 0
    finish(s,1)
    assert s.gate and s.progress == 0  # applies only to next iteration
    finish(s,2,.9,failures=0)
    finish(s,3,.9,failures=0)
    assert (s.tracking_reference,s.failure_reference) == (.9,0)
    finish(s,4,.7,failures=2)
    finish(s,5,.7,failures=2)
    assert (s.tracking_reference,s.failure_reference) == (.9,0)
    assert not s.gate and s.reason == s.TRACKING
    progress = s.progress
    finish(s,6,.9,failures=1)
    finish(s,7,.9,failures=1)
    assert not s.gate and s.reason == s.FAILURES and s.progress == progress


def test_weighted_evidence_missing_window_and_absolute_allowance():
    s = InterventionCurriculum(config(intervention_failure_allowance=.02))
    finish(s,0,.5,count=1,failures=0,exposure=1)
    finish(s,1,1.,count=3,failures=0,exposure=9)
    assert s.tracking_mean == .875 and s.failure_reference == 0
    finish(s,2,.9,failures=0,exposure=1)
    finish(s,3,.9,failures=1,exposure=99)
    assert s.gate and s.failure_rate == .01  # NOT mean of per-iteration rates
    before = s.progress
    finish(s,4,None)
    finish(s,5,.9)
    assert s.progress == before and not s.gate
    finish(s,6,.9,failures=0)
    finish(s,7,.9,failures=0)
    assert s.gate and s.progress == before
    finish(s,9,.9,failures=0)  # missing iteration invalidates window
    finish(s,10,.9,failures=0)
    assert not s.gate


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),None])
def test_nonfinite_evidence_pauses(bad):
    s = InterventionCurriculum(config())
    finish(s,0);finish(s,1)
    finish(s,2,bad)
    assert not s.gate and s.progress == 0


def test_checkpoint_partial_window_and_old_checkpoint_fresh_evidence():
    cfg = config()
    original = QPCurriculum(cfg, actor_ramp_iterations=3)
    for i in range(6):
        original.begin(i)
        finish(original.intervention,i)
    state = deepcopy(original.state_dict())
    restored = QPCurriculum(cfg)
    restored.load_state_dict(state)
    for i in range(6,20):
        assert original.begin(i) == restored.begin(i)
        for obj in (original,restored):
            finish(obj.intervention,i,score=.9 if i<12 else .6)
        assert original.state_dict() == restored.state_dict()
    state.pop('intervention')
    older = QPCurriculum(cfg, actor_ramp_iterations=3)
    older.load_state_dict(state)
    assert older.intervention.begin(10000) == (1,0,0)
    finish(older.intervention,10000)
    assert not older.intervention.gate and older.intervention.progress == 0
    finish(older.intervention,10001)
    assert older.intervention.gate and older.intervention.progress == 0


@pytest.mark.parametrize('n',[0,1,2,3,7,10,4096])
@pytest.mark.parametrize('progress',[0,.25,.5,.75,1])
def test_sampling_counts_anchor_and_owned_rng(n,progress):
    probabilities = intervention_probabilities(progress)
    kwargs = dict(intervention_probabilities=probabilities)
    g = torch.Generator().manual_seed(27)
    selected,anchor = qp_interval_selection('curriculum_substeps',n,4,'cpu',generator=g,**kwargs)
    masks = torch.stack([qp_substep_mask('curriculum_substeps',k,selected) for k in range(4)])
    counts = masks.sum(0)
    assert masks[0].all() and masks.gather(0,anchor[None]).all()
    for k,p in zip((2,3,4),probabilities):
        assert abs((counts==k).sum().item()-n*p) < 1.000001
    again = qp_interval_selection('curriculum_substeps',n,4,'cpu',
        generator=torch.Generator().manual_seed(27),**kwargs)
    assert all(torch.equal(a,b) for a,b in zip((selected,anchor),again))


def test_uniform_anchor_not_success_selected_and_control_replay():
    selected,anchor = qp_interval_selection('curriculum_substeps',30000,4,'cpu',
        intervention_probabilities=(.5,.3,.2),generator=torch.Generator().manual_seed(1))
    masks = torch.stack([qp_substep_mask('curriculum_substeps',k,selected) for k in range(4)])
    counts = masks.sum(0)
    for k in (2,3,4):
        assert abs((anchor[counts==k]==0).float().mean().item()-1/k)<.02
    task,heads,qp,_,solves,q,v,quat,_ = fixture('curriculum_substeps',n=10)
    task._qp_intervention_probabilities = (.5,.3,.2)
    task._begin_qp_interval()
    anchor = task._qp_sampled_substep_index.clone()
    calls = heads.calls
    for k in range(4):
        q.fill_(.001*k); v.fill_(.0001*k)
        task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(10,6))
    assert solves.sum() == 27 and heads.calls == calls
    assert task._qp_sampled_transition['sampled_qp_valid'].all()
    torch.testing.assert_close(task._qp_sampled_transition['sampled_qp_q'][:,7:],
        (.001*anchor.float())[:,None].expand(-1,12))
    assert torch.isfinite(task.simulator._torques).all()


def test_runner_snapshot_collects_when_other_curricula_disabled():
    from types import SimpleNamespace as NS
    from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT
    task, _, qp, *_ = fixture('curriculum_substeps', n=4)
    qp.cfg = config(objective_curriculum_enabled=False)
    runner = NS(alg=NS(hard_pact_qp=qp, qp_enabled_at_iteration=lambda: True,
                       actor_physics_warmup_iterations=5),
                env=NS(set_hard_pact_qp_enabled=lambda _: None))
    OnPolicyRunnerPACT._set_hard_pact_qp_iteration(runner,5)
    s = runner.qp_curriculum.intervention
    assert s.ramp_end == 5 and runner.env._qp_tracking_collect
    assert runner.env._qp_intervention_probabilities == (1,0,0)
    finish(s,5); finish(s,6); finish(s,7)
    OnPolicyRunnerPACT._set_hard_pact_qp_iteration(runner,8)
    assert runner.env._qp_intervention_probabilities == intervention_probabilities(s.progress)
    assert qp.cfg.contact_acceleration_weight == runner.qp_curriculum.cfg.contact_acceleration_weight


def test_old_contract_migration_only_changes_schedule():
    from rsl_rl.algorithms.hard_pact_intervention_curriculum import legacy_intervention_contract
    from legged_gym.envs.go2.go2_hard_pact.deployment import qp_update_contract
    new = qp_update_contract('curriculum_substeps',4,qp_config=config())
    old = qp_update_contract('mixed_substeps',4,qp_config=config(qp_update_mode='mixed_substeps'))
    old['solver_and_objective_settings'] = {k:v for k,v in old['solver_and_objective_settings'].items()
                                            if not k.startswith('intervention_')}
    saved = deepcopy(old)
    assert legacy_intervention_contract(old,new) == new and old == saved
    old['solver_and_objective_settings']['torque_tracking_weight'] += 1
    assert legacy_intervention_contract(old,new) != new


def test_intervention_full_checkpoint_resume_and_rng(tmp_path):
    from test_training_state_transfer import hard
    from rsl_rl.utils import hard_pact_checkpoint as hc
    cfg = config()
    source = hard()
    source.alg.qp_config = cfg
    source.qp_curriculum = QPCurriculum(cfg,3)
    for i in range(8): finish(source.qp_curriculum.intervention,i)
    source.current_learning_iteration = 8
    saved_schedule = deepcopy(source.qp_curriculum.intervention.state_dict())
    path = tmp_path/'source.pt'
    source.save(path)
    expected = qp_interval_selection('curriculum_substeps',17,4,'cpu',
        intervention_probabilities=source.qp_curriculum.intervention.begin(8))
    target = hard()
    target.alg.qp_config = cfg
    hc.load(target,path)
    assert target.current_learning_iteration == 8
    assert target.qp_curriculum.intervention.state_dict() == saved_schedule
    actual = qp_interval_selection('curriculum_substeps',17,4,'cpu',
        intervention_probabilities=target.qp_curriculum.intervention.begin(8))
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))


@pytest.mark.parametrize('kw', [dict(intervention_ramp_iterations=0),
    dict(intervention_window_iterations=0),dict(intervention_settling_iterations=-1),
    dict(intervention_failure_allowance=-1),dict(intervention_tracking_ratio=1.1)])
def test_invalid_config(kw):
    with pytest.raises(ValueError): InterventionCurriculum(config(**kw))
