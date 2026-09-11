"""CPU backward smoke tests; simulator methods run against a deterministic fake robot."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from rsl_rl.algorithms.ppo_pact import PPO_PACT
from rsl_rl.algorithms.ppo_abl3 import PPO_ABL3
from rsl_rl.algorithms.ppo_pact_pos import PPO_PACT_Pos
from rsl_rl.modules import ActorCritic_PACT, ActorCritic_PACT_Pos, ContextDecoder
from rsl_rl.modules.grf_transition import capture_substep, commanded_torque, reconstruction
from rsl_rl.modules.grf_checkpoint import load_grf_decoders

ROOT = Path(__file__).resolve().parents[1]


def method(path, name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    ns = dict(torch=torch, capture_substep=capture_substep)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)
    return ns[name]


def metadata(batch=4):
    t = torch.zeros(batch, 86)
    t[:, 12:36] = 1
    t[:, 48:60] = -100
    t[:, 60:72] = 100
    t[:, 84:] = 1
    return t


def algorithm(kind, mode='mse'):
    actor_type = ActorCritic_PACT_Pos if kind is PPO_PACT_Pos else ActorCritic_PACT
    actor = actor_type(57, 288, 12, [16, 8], [16, 8], 57, 16, 16, [16, 8], 'elu')
    return kind(actor, ContextDecoder(32, [16]*3, 276), 288,
                grf_decoder_network=ContextDecoder(44, [16]*3, 12),
                aligned_grf_transition=True, grf_reconstruction_mode=mode,
                dof_tau_observation_scale=.05, num_encoder_epochs=1)


@pytest.mark.parametrize('task,sim_name', [('pact', 'pact'), ('abl3', 'pact_nopinn'), ('pact_pos', 'pact_pos')])
def test_real_queue_and_pd_substep_capture(task, sim_name):
    cfg = NS(control=NS(decimation=3, action_scale=.5, torque_scale=2),
             normalization=NS(clip_actions=2.), domain_rand=NS(randomize_ctrl_delay=True))
    env = NS(cfg=cfg, device='cpu', num_envs=2, action_delay=torch.tensor([0, 1]),
             action_queue=torch.ones(2, 2, 24)*.3,
             actions=torch.zeros(2, 24), last_actions=torch.zeros(2, 24), llast_actions=torch.zeros(2, 24))
    if task == 'pact_pos':
        env.action_queue = env.action_queue[..., :12].clone()
        for field in ('actions', 'last_actions', 'llast_actions'):
            setattr(env, field, getattr(env, field)[..., :12].clone())
    width = env.actions.shape[-1]
    queued = method(f'legged_gym/envs/go1/go1_{task}/go1_{task}.py', '_pre_sim_step')(env, torch.ones(2, width)*3)
    torch.testing.assert_close(queued[0], torch.full((width,), 2.))
    torch.testing.assert_close(queued[1], torch.full((width,), .3))
    sim = NS(_cfg=cfg, _device='cpu', _num_envs=2, _control_dt=.03,
             _dof_pos=torch.zeros(2,12), _dof_vel=torch.zeros(2,12), _default_dof_pos=torch.zeros(12),
             _p_gains=torch.ones(12)*2, _d_gains=torch.ones(12),
             _kp_scale=torch.tensor([[.8],[1.2]]), _kd_scale=torch.tensor([[1.1],[.7]]),
             _motor_strength=torch.tensor([[.9],[1.3]]), feedback_tau_weight=torch.tensor([[.6],[.8]]),
             feedforward_tau_weight=torch.tensor([[.4],[.2]]), _dof_indices=list(range(12)),
             _grf_current_causal=env.action_delay==0)
    for name in ('base_lin_vel','base_ang_vel','feet_vel','base_world_lin_vel','base_world_ang_vel'):
        setattr(sim, '_'+name, torch.zeros(2,3))
        setattr(sim, '_last_'+name, torch.zeros(2,3))
    sim._last_dof_vel=torch.zeros(2,12)
    robot_pos, robot_vel = torch.zeros(2,12), torch.zeros(2,12)
    applied=[]
    def advance():
        robot_vel.add_(applied[-1]*.01)
        robot_pos.add_(robot_vel*.01)
    sim._robot=NS(get_dofs_force_range=lambda _: (-torch.ones(12), torch.ones(12)),
                  get_vel=lambda: robot_vel[:,:3], get_ang=lambda: robot_vel[:,3:6],
                  control_dofs_force=lambda tau, _: applied.append(tau.clamp(-1,1).clone()),
                  get_dofs_position=lambda _: robot_pos, get_dofs_velocity=lambda _: robot_vel)
    sim._scene=NS(step=advance)
    torque_fn=method(f'legged_gym/simulator/genesis_simulator_{sim_name}.py', '_compute_torques')
    sim._compute_torques=lambda actions: torque_fn(sim, actions)
    method(f'legged_gym/simulator/genesis_simulator_{sim_name}.py', 'step')(sim, queued)
    torch.testing.assert_close(sim._grf_transition[:,:12], applied[-1])
    replay_actions = torch.cat((queued, torch.zeros_like(queued)), -1) if width == 12 else queued
    torch.testing.assert_close(commanded_torque(replay_actions, sim._grf_transition), applied[-1])
    torch.testing.assert_close(sim._grf_transition[:,72:84], applied[-1])
    assert sim._grf_transition[:,84].tolist()==[1,0]
    # A reset cannot mutate the copied rollout torque/state.
    saved=sim._grf_transition.clone()
    sim._dof_pos.zero_(); sim._motor_strength.zero_()
    torch.testing.assert_close(commanded_torque(replay_actions, saved), applied[-1])


@pytest.mark.parametrize('mode', ['mse','huber'])
def test_mask_scaling_and_empty(mode):
    pred=torch.tensor([[2.]*12, [1.e6]*12], requires_grad=True)
    loss,mse=reconstruction(pred, torch.zeros_like(pred), torch.tensor([[1.],[0.]]), mode, .5)
    assert mse.item()==4.
    assert loss.item()==(4. if mode=='mse' else .875)
    loss.backward(); assert pred.grad[1].count_nonzero()==0
    empty,_=reconstruction(pred, torch.zeros_like(pred), torch.zeros(2,1), mode, .5)
    assert empty.item()==0 and torch.isfinite(empty)
    tau=torch.ones(2,12,requires_grad=True)*20
    scaled=PPO_PACT._grf_decoder_input(torch.zeros(2,32),tau,.05)
    torch.testing.assert_close(scaled[:,-12:],torch.ones(2,12))


@pytest.mark.parametrize('kind', [PPO_PACT,PPO_ABL3,PPO_PACT_Pos])
def test_supervised_gradient_routing(kind):
    alg=algorithm(kind)
    applied=torch.randn(4,12,requires_grad=True)
    result=alg._compute_vae_loss(torch.randn(4,57),torch.randn(4,12),torch.randn(4,288),torch.randn(4,16),torch.ones(4,1),applied)
    result[3].backward()  # GRF-only supervised objective
    assert applied.grad is None
    assert alg.actor_critic.act_pos_out.weight.grad is None
    assert alg.actor_critic.act_tau_out.weight.grad is None
    assert alg.actor_critic.context_encoder.ce_in.weight.grad.norm()>0
    assert alg.grf_decoder.dec_out.weight.grad.norm()>0
    assert alg.grf_decoder.dec_in.in_features==44
    assert alg.decoder.dec_out.out_features==276
    assert alg.actor_critic.critic[0].in_features==288
    if kind in (PPO_ABL3, PPO_PACT_Pos):
        assert not hasattr(alg,'_compute_PINN_loss')
        opt = alg.act_optimizer.optimizer if kind is PPO_PACT_Pos else alg.act_optimizer
        ppo_ids={id(p) for g in opt.param_groups for p in g['params']}
        assert not ppo_ids.intersection(id(p) for p in alg.grf_decoder.parameters())


def test_pinn_joint_path_and_delayed_mask():
    alg=algorithm(PPO_PACT)
    alg.pinn_grf_reconstruction_mse_threshold=float('inf')
    alg.grf_transition_batch=metadata()
    alg.grf_transition_batch[1,84]=0
    actions=torch.randn(4,24,requires_grad=True)*.1
    actions.retain_grad()
    history=torch.randn(4,57)
    jac=torch.ones(4,18,12)
    # Prove the torque -> GRF path independently of the residual direct torque term.
    torque=commanded_torque(actions,alg.grf_transition_batch)
    selected,_,gate=alg._select_pinn_contact_forces(history,torch.zeros(4,12),torch.ones(4,1),torque,jac,torch.zeros(4,18))
    assert gate
    grad=torch.autograd.grad(selected.sum(),actions)[0]
    assert grad.norm()>0
    actions.grad = None
    # Bias positive contacts so the existing contact weighting is active.
    with torch.no_grad(): alg.grf_decoder.dec_out.bias.fill_(1)
    valid=torch.tensor([[1.],[1.],[0.],[1.]])
    loss=alg._compute_PINN_loss(actions,torch.zeros(4,57),history,None,None,None,None,
        torch.randn(4,6),torch.eye(18).repeat(4,1,1),torch.randn(4,18),torch.ones(4,18),
        jac,torch.zeros(4,12),valid,None,None,None,.02,1.)
    loss.backward()
    assert torch.isfinite(loss) and actions.grad[0].norm()>0
    assert actions.grad[1:3].count_nonzero()==0  # queued delay and reset
    assert all(p.grad is None for p in alg.grf_decoder.parameters())
    assert alg.actor_critic.context_encoder.ce_in.weight.grad.norm()>0


@pytest.mark.parametrize('kind', [PPO_PACT,PPO_ABL3,PPO_PACT_Pos])
@pytest.mark.parametrize('mode', ['mse','huber'])
def test_short_rollout_backward_update(kind,mode,monkeypatch):
    torch.manual_seed(4)
    alg=algorithm(kind,mode)
    if kind is PPO_PACT_Pos:
        # Existing CUDA calls only time work; this smoke test executes on CPU.
        monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
        monkeypatch.setattr(alg, '_nominal_torque_from_action', lambda *args: pytest.fail('Recomputed supervised torque'))
    args=(4,2,[57],[288],[288],[57],[12 if kind is PPO_PACT_Pos else 24],[16],[12])
    alg.init_storage(*args,*([[18]] if kind is PPO_PACT else []))
    for _ in range(2):
        obs,hist,critic=torch.randn(4,57),torch.randn(4,57),torch.randn(4,288)
        with torch.no_grad():
            alg.act(obs,critic,hist,*([obs,hist,obs,hist] if kind is PPO_PACT else []))
        data=metadata();data[:,:12]=torch.randn(4,12);data[1,84]=0
        info={'grf_transition':data,'time_outs':torch.tensor([0,0,1,0])}
        args=(torch.ones(4),torch.tensor([0,0,1,0]),info,torch.randn(4,12),critic,torch.randn(4,16))
        if kind is PPO_PACT:
            alg.process_env_step(*args,torch.ones(4,18),torch.ones(4,18,12),torch.eye(18).repeat(4,1,1),torch.randn(4,18),torch.randn(4,6))
        else: alg.process_env_step(*args)
    alg.compute_returns(torch.randn(4,288))
    before=alg.grf_decoder.dec_out.weight.detach().clone()
    if kind is PPO_PACT:
        alg.pinn_weight=.01
        alg.pinn_grf_reconstruction_mse_threshold=float('inf')
        def forbidden_recompute(*args):
            raise AssertionError("Aligned GRF paths must use the cached PD law, not legacy action transforms")
        result=alg.update(forbidden_recompute,forbidden_recompute,.02,0,torch.zeros(12),1.)
    elif kind is PPO_PACT_Pos:
        result=alg.update(lambda a:(a[:,:12],a[:,12:]),lambda q,p,v:q-p-v,.02,0,torch.zeros(12),1.)
    else: result=alg.update()
    assert all(torch.isfinite(torch.tensor(x)) for x in result)
    assert not torch.equal(before,alg.grf_decoder.dec_out.weight)
    assert alg.last_grf_mse>=0


def test_abl3_checkpoint_migration_and_roundtrip():
    alg=algorithm(PPO_ABL3)
    old=ContextDecoder(32,[16]*3,288).state_dict()
    with pytest.warns(UserWarning,match='Legacy ABL3'):
        load_grf_decoders(alg,{'decoder_state_dict':old})
    torch.testing.assert_close(alg.decoder.dec_out.weight,torch.cat((old['dec_out.weight'][:61],old['dec_out.weight'][73:])))
    ckpt={'decoder_state_dict':alg.decoder.state_dict(),'grf_decoder_state_dict':alg.grf_decoder.state_dict()}
    restored=algorithm(PPO_ABL3);load_grf_decoders(restored,ckpt)
    for key,val in alg.grf_decoder.state_dict().items(): torch.testing.assert_close(restored.grf_decoder.state_dict()[key],val)
    broken=dict(ckpt);broken['grf_decoder_state_dict']=ContextDecoder(32,[16]*3,12).state_dict()
    with pytest.raises(RuntimeError): load_grf_decoders(restored,broken)


@pytest.mark.parametrize('kind', [PPO_PACT,PPO_ABL3,PPO_PACT_Pos])
def test_runner_checkpoint_optimizer_and_modes(kind,tmp_path):
    from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT
    from rsl_rl.runners.abl3_runner import OnPolicyRunnerABL3
    from rsl_rl.runners.pact_pos_runner import OnPolicyRunnerPACTPos
    runner_type={PPO_PACT:OnPolicyRunnerPACT,PPO_ABL3:OnPolicyRunnerABL3,PPO_PACT_Pos:OnPolicyRunnerPACTPos}[kind]
    runner=runner_type.__new__(runner_type)
    runner.alg=algorithm(kind);runner.device='cpu';runner.current_learning_iteration=7
    # Populate Adam moments so the test exercises state, not only group layout.
    opt=runner.alg.grf_decoder_optimizer
    sum(p.square().sum() for p in runner.alg.grf_decoder.parameters()).backward();opt.step()
    runner.alg.test_mode()
    assert not runner.alg.grf_decoder.training and not runner.alg.decoder.training
    runner.alg.train_mode()
    assert runner.alg.grf_decoder.training and runner.alg.decoder.training
    path=tmp_path/'model.pt';runner.save(path)
    restored=runner_type.__new__(runner_type)
    restored.alg=algorithm(kind);restored.device='cpu'
    restored.load(path)
    assert len(restored.alg.grf_decoder_optimizer.state)==len(opt.state)
    for key,val in runner.alg.grf_decoder.state_dict().items():
        torch.testing.assert_close(restored.alg.grf_decoder.state_dict()[key],val)


def test_go1_configs_match_grf_head():
    configs=[]
    for task in ('pact','abl3','pact_pos'):
        tree=ast.parse((ROOT/f'legged_gym/envs/go1/go1_{task}/go1_{task}_config.py').read_text())
        policy=next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name=='policy')
        names={'cenet_dec_input_dim','cenet_dec_out_dim','grf_dec_input_dim','grf_dec_layers',
               'grf_dec_out_dim','grf_torque_observation_scale','privileged_grf_start_index'}
        values={}
        for node in policy.body:
            if isinstance(node,ast.Assign) and node.targets[0].id in names:
                values[node.targets[0].id]=eval(compile(ast.Expression(node.value),'config','eval'),values)
        configs.append({k:values[k] for k in names})
    assert configs[0]==configs[1]==configs[2]
    assert configs[0]['grf_dec_input_dim']==44
    assert configs[0]['grf_torque_observation_scale']==.01


def test_pact_pos_legacy_checkpoint_keeps_new_grf_head(tmp_path):
    from rsl_rl.runners.pact_pos_runner import OnPolicyRunnerPACTPos
    runner=OnPolicyRunnerPACTPos.__new__(OnPolicyRunnerPACTPos)
    runner.alg=algorithm(PPO_PACT_Pos);runner.device='cpu';runner.current_learning_iteration=0
    path=tmp_path/'legacy_pos.pt';runner.save(path)
    checkpoint=torch.load(path,weights_only=False)
    del checkpoint['grf_decoder_state_dict']
    del checkpoint['grf_decoder_opt_state_dict']
    old=ContextDecoder(32,[16]*3,288).state_dict()
    checkpoint['decoder_state_dict']=old
    torch.save(checkpoint,path)
    before={k:v.clone() for k,v in runner.alg.grf_decoder.state_dict().items()}
    with pytest.warns(UserWarning,match='Legacy PACT-Pos'):
        runner.load(path)
    torch.testing.assert_close(runner.alg.decoder.dec_in.weight,old['dec_in.weight'])
    assert not runner.alg.decoder_optimizer.state
    assert not runner.alg.grf_decoder_optimizer.state
    for key,val in before.items(): torch.testing.assert_close(runner.alg.grf_decoder.state_dict()[key],val)
