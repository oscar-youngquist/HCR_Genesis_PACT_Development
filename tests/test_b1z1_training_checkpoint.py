"""CPU learning-state round trips; no simulator or rollout is initialized."""
from copy import deepcopy
from types import SimpleNamespace as NS
import random

import numpy as np
import pytest
import torch

from rsl_rl.utils import b1z1_training_checkpoint as ck
from rsl_rl.runners.b1z1_pact_runner import B1Z1PACTRunner


class State:
    def __init__(self): self.value = {'progress': .7}
    def state_dict(self): return deepcopy(self.value)
    def load_state_dict(self, value): self.value = deepcopy(value)


def runner():
    r = B1Z1PACTRunner.__new__(B1Z1PACTRunner)
    r.device = 'cpu'
    r.actor_critic = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    r.privileged_decoder = torch.nn.Linear(2, 1)
    r.alg = NS(actor_optimizer=NS(optimizer=torch.optim.Adam(r.actor_critic[0].parameters(), lr=.003)),
        auxiliary_optimizer=torch.optim.Adam(r.actor_critic[1].parameters(), lr=.004),
        decoder_optimizer=torch.optim.Adam(r.privileged_decoder.parameters(), lr=.005),
        storage=NS(step=0), transition=NS(actions=None), learning_rate=.003,
        pinn_weight=.4, pinn_updates=71, _spec_u={'layer': torch.ones(3)},
        force_ema=.2, force_gate_active=True, force_gate_count=7, force_metric_emas={'ee': .1},
        force_blend_start_ema=.3, current_entropy_coef=.005, kl_controller=State())
    sim = NS(feedforward_tau_weight_clean=torch.ones(2,1), feedback_tau_weight_clean=torch.ones(2,1))
    r.env = NS(cfg=NS(seed=1, env=NS(num_envs=2)), simulator=sim,
        command_ranges={'x': [-1., 1.]}, reward_scales={'tracking': .03},
        tradeoff_step_ctr=torch.ones(2,1)*5, common_step_counter=40,
        _staged_force_curriculum=State())
    r.cfg = {'policy': {'conditioning_mode': 'film'}, 'algorithm': {'lr': .001},
             'runner': {'max_iterations': 100, 'num_steps_per_env': 24}}
    r._training_contract = ck.contract(r)
    r.current_learning_iteration=10; r.total_timesteps=480; r.total_time=15.
    r._update_in_progress=False; r._resume_state=None
    return r


def update(r):
    for opt in ck.optimizers(r).values(): opt.zero_grad()
    x=torch.randn(4,2)
    loss=r.actor_critic(x).square().mean()+r.privileged_decoder(x).square().mean()
    loss.backward()
    for opt in ck.optimizers(r).values(): opt.step()
    return loss.detach()


def test_full_roundtrip_next_optimizer_update_and_rng(tmp_path):
    source=runner(); update(source)
    source._completed_rewards=[2.,3.]; source._completed_lengths=[17,22]
    path=tmp_path/'model.pt'; source.save(path,iteration=11)
    expected_py=random.random(); expected_np=np.random.rand()
    expected=update(source)
    target=runner(); target.load(path)
    assert target.current_learning_iteration==11
    assert (target.total_timesteps,target.total_time)==(480,15.)
    assert target._completed_lengths==[17,22] and target._completed_rewards==[2.,3.]
    assert target.alg.learning_rate==.003 and target.alg.pinn_updates==71
    torch.testing.assert_close(target.env.tradeoff_step_ctr,source.env.tradeoff_step_ctr)
    ck.restore_rng(target,target._resume_state['rng'])
    assert random.random()==expected_py and np.random.rand()==expected_np
    torch.testing.assert_close(update(target),expected,rtol=0,atol=0)
    for a,b in zip(source.actor_critic.parameters(),target.actor_critic.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_contract_preflight_does_not_mutate_and_allows_runtime_budget(tmp_path):
    source=runner(); path=tmp_path/'model.pt'; source.save(path)
    target=runner(); target.cfg['runner']['max_iterations']=200
    target._training_contract=ck.contract(target); target.load(path)
    target=runner(); target.cfg['policy']['conditioning_mode']='concat'
    target._training_contract=ck.contract(target)
    before=deepcopy(target.actor_critic.state_dict())
    with pytest.raises(ValueError,match='contract'): target.load(path)
    for k,v in before.items(): torch.testing.assert_close(v,target.actor_critic.state_dict()[k])


def test_legacy_and_weights_only_are_explicit(tmp_path):
    source=runner(); path=tmp_path/'model.pt'; source.save(path)
    target=runner(); target.current_learning_iteration=0
    target.total_timesteps=0; target.alg.learning_rate=.009
    assert target.load(path,load_optimizer=False)==0
    assert target.current_learning_iteration==0 and target.total_timesteps==0
    assert target.alg.learning_rate==.009 and target._resume_state is None
    data=torch.load(path,weights_only=False); data.pop('training_state'); torch.save(data,path)
    with pytest.warns(RuntimeWarning,match='Legacy partial'): target.load(path)


def test_boundary_and_atomic_failure_preserve_previous_file(tmp_path,monkeypatch):
    source=runner(); path=tmp_path/'model.pt'; source.save(path); before=path.read_bytes()
    source.alg.storage.step=1
    with pytest.raises(RuntimeError,match='completed'): source.save(path)
    source.alg.storage.step=0; source._update_in_progress=True
    with pytest.raises(RuntimeError,match='completed'): source.save(path)
    source._update_in_progress=False
    def fail(*args,**kwargs): raise OSError('disk full')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError): source.save(path)
    assert path.read_bytes()==before
    assert list(tmp_path.iterdir())==[path]


def test_resume_resets_history_once_without_resetting_progress(tmp_path):
    source=runner(); path=tmp_path/'model.pt'; source.save(path)
    target=runner(); target.load(path)
    calls=[]
    def reset():
        assert target.env._restoring_training_episodes
        calls.append('environment')
        target.env.tradeoff_step_ctr.zero_()
        target.env.common_step_counter=0
        target.env._staged_force_curriculum.value={'progress': 0.}
    target.env.reset=reset
    target.actor_critic.reset=lambda: calls.append('history')
    target.action_replay=NS(queue=torch.ones(2,3))
    target._resume_fresh_episodes()
    target._resume_fresh_episodes()
    assert calls==['environment','history']
    assert not target.env._restoring_training_episodes
    assert target.action_replay.queue is None
    assert target.env.common_step_counter==40
    assert target.env.tradeoff_step_ctr.eq(5).all()
    assert target.env._staged_force_curriculum.value=={'progress': .7}


def test_corrupted_moments_rejected_before_model_mutation(tmp_path):
    source=runner(); update(source); path=tmp_path/'model.pt'; source.save(path)
    data=torch.load(path,weights_only=False)
    moment=next(iter(data['actor_optimizer']['state'].values()))
    moment['exp_avg']=torch.zeros(999)
    torch.save(data,path)
    target=runner(); before=deepcopy(target.actor_critic.state_dict())
    with pytest.raises(ValueError,match='moment'): target.load(path)
    for k,v in before.items(): torch.testing.assert_close(v,target.actor_critic.state_dict()[k])
