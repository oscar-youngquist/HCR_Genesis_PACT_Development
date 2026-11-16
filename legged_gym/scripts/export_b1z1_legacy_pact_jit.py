#!/usr/bin/env python3
"""Export the saved pre-compliance contract without importing environments/simulators."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_class():
    # Avoid rsl_rl.modules.__init__: importing the training stack is unnecessary.
    package = types.ModuleType('_legacy_pact_reference')
    package.__path__ = [str(ROOT/'rsl_rl/modules')]
    sys.modules[package.__name__] = package
    return load_file(package.__name__+'.actor_critic_b1z1_pact',
                     ROOT/'rsl_rl/modules/actor_critic_b1z1_pact.py').ActorCriticB1Z1PACT


def value(tree, path):
    """Read explicit saved constants, never execute an archived Python config."""
    parts = path.split('.')
    scope = tree
    for name in parts[:-1]:
        scope = next(n for n in scope.body if isinstance(n, ast.ClassDef) and n.name == name)
    node = next(n.value for n in reversed(scope.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == parts[-1] for t in n.targets))
    def literal(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Mult)):
            a, b = literal(n.left), literal(n.right)
            return a+b if isinstance(n.op, ast.Add) else a*b
        return ast.literal_eval(n)
    return literal(node)


def build(checkpoint, config):
    tree = ast.parse(config.read_text())
    env = lambda name: value(tree, 'B1Z1PACTCfg.'+name)
    policy = lambda name: value(tree, 'B1Z1PACTCfgPPO.policy.'+name)
    if (env('env.num_observations'), env('env.num_obs_hist'), env('env.num_actions')) != (81, 32, 17):
        raise ValueError('Only legacy 81-D / 32-frame / 17+17-action checkpoints are supported')
    if not policy('film_v02_enabled') or policy('film_history_steps') != 5 or policy('conditioning_mode') != 'film':
        raise ValueError('This exporter requires the saved five-frame latent FiLM V02 contract')
    if policy('activation') != 'elu':
        raise ValueError('This legacy export has only been verified for ELU')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = state['model_state_dict']
    actor = reference_class()(num_actor_obs=81, num_critic_obs=weights['critic.0.weight'].shape[1],
        num_actions=17, history_dim=2592, latent_dim=policy('cenet_latent_dim'),
        actor_layers=policy('actor_layers'), critic_layers=policy('critic_layers'),
        context_layers=policy('cenet_enc_layers'), explicit_decoder_layers=policy('explicit_decoder_layers'),
        force_decoder_layers=policy('force_decoder_layers'), grf_decoder_layers=policy('grf_decoder_layers'),
        film_hidden_dim=policy('film_hidden_dim'), film_v02_enabled=True, film_history_steps=5,
        film_base_error_soft_scale=policy('film_base_error_soft_scale'),
        film_ee_error_soft_scale=policy('film_ee_error_soft_scale'), activation='elu')
    actor.load_state_dict(weights, strict=True)
    actor.eval().requires_grad_(False)
    names = env('asset.dof_names')
    def gains(kind):
        mapping = env('control.'+kind)
        result = []
        for name in names:
            matches = [v for k, v in mapping.items() if k in name]
            if len(matches) != 1:
                raise ValueError('Ambiguous/missing gain for '+name)
            result.append(matches[0])
        return result
    scales = env('normalization.obs_scales.lin_vel'), env('normalization.obs_scales.ang_vel')
    override = env('control.torque_scale_overrides')
    settings = dict(base_soft=policy('film_base_error_soft_scale'), ee_soft=policy('film_ee_error_soft_scale'),
        clip_obs=env('normalization.clip_observations'), clip_action=env('normalization.clip_actions'),
        position_scale=env('control.action_scale'), angular_scale=scales[1],
        q_scale=env('normalization.obs_scales.dof_pos'), qd_scale=env('normalization.obs_scales.dof_vel'),
        dt=env('control.dt'), cycle=env('rewards.cycle_time'),
        q_default=[env('init_state.default_joint_angles')[n] for n in names], kp=gains('stiffness'), kd=gains('damping'),
        ff_scale=[override.get(n, env('control.torque_scale')) for n in names[:17]],
        command_scale=[scales[0], scales[0], scales[1]] + [env('normalization.obs_scales.ee_sphe_'+n+'_cmd') for n in ('radius','pitch','yaw')],
        walking_threshold=[env('commands.'+n+'_clip') for n in ('lin_vel_x','lin_vel_y','ang_vel_yaw')])
    cls = load_file('_legacy_pact_deployment', ROOT/'rsl_rl/b1z1_legacy_deployment.py').LegacyPACTDeployment
    deployment = cls(actor, settings).eval().requires_grad_(False)
    metadata = dict(schema='b1z1_legacy_pact_jit_v1', checkpoint_iteration=state.get('iteration'),
        joint_names=names, learned_joint_names=names[:17], held_joint_names=names[17:],
        foot_names=env('asset.foot_name'), settings=settings,
        sphere_center={k:env('goal_ee.sphere_center.'+k) for k in ('x_offset','y_offset','z_invariant_offset')},
        action_order='position[17],feedforward[17]; previous issued CLIPPED action, before delay',
        history_order='oldest to newest, current frame included; zero padding at reset',
        observation_fields=[['roll_pitch_rad',0,2],['omega_body_scaled',2,5],['q_minus_default_scaled',5,22],
                            ['qd_scaled',22,39],['phase_sin_cos',39,41],['previous_issued_action',41,75],['commands_scaled',75,81]],
        explicit_order='vx,vy,yaw_rate,radius,elevation,azimuth,contact_probability[4],foot_height[4]',
        force_order='torso_force[3],torso_moment[3],ee_force[3]; heading frame, normalized training outputs',
        force_scale=env('normalization.obs_scales.ee_force'), base_force_scale=env('normalization.obs_scales.base_force'),
        grf_scale=env('normalization.obs_scales.grf'), height_scale=env('normalization.obs_scales.height_measurements'),
        no_differential_compliance=True, observation_noise_in_export=False,
        grf_semantics='Legacy checkpoint prediction; not retrained with later tangential-GRF supervision.',
        grf_frame='Successor-heading-aligned XYZ; rotate by successor yaw for world-frame diagnostics.',
        foot_height_units='meters (unscaled)',
        clipping='Training execution used 1.1 * rated torque limits AFTER motor scaling; supply execution_limits explicitly.',
        deployment_nominal='PD/FF weights and motor/Kp/Kd multipliers all 1; optional delay=0. No randomization in export.')
    return actor, deployment, metadata


@torch.no_grad()
def validate(actor, module):
    torch.manual_seed(70000)
    errors = []
    for batch in (1, 3):
        actor.reset()
        hist, tracking, prev = module.initial_state(batch)
        for step in range(40):
            reset = torch.zeros(batch, dtype=torch.bool)
            if step in (0, 19):
                reset[0] = True
                actor.reset(reset)
                hist[reset] = 0
                prev[reset] = 0
            q = module.q_default.expand(batch, -1) + .1*torch.randn(batch,19)
            obs = module.build_observation(.1*torch.randn(batch,2), torch.randn(batch,3), q,
                torch.randn(batch,19), torch.rand(batch), prev, torch.randn(batch,6)*.2)
            expected_hist = torch.cat((hist[:,81:],obs),-1)
            expected = actor.act_inference(obs.clamp(-module.clip_obs,module.clip_obs), expected_hist)
            raw, issued, _, _, hist, tracking = module.step(obs, hist, tracking, reset)
            torch.testing.assert_close(raw, expected, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(hist, expected_hist)
            torch.testing.assert_close(tracking, actor._film_history)
            errors.append(float((raw-expected).abs().max()))
            prev = issued
        z, explicit, force = module.estimate(hist)
        context = actor.decode_context(actor.context_encoder.forward_inf(hist))
        torch.testing.assert_close(z, context['z'])
        torch.testing.assert_close(explicit, context['explicit_condition'])
        torch.testing.assert_close(force, torch.cat((context['base_wrench'],context['ee_force']),-1))
        tau = torch.randn(batch,19)
        torch.testing.assert_close(module.predict_grf(hist,tau),actor.predict_grf(context,tau))
    return dict(sequence_steps=80, batch_sizes=[1,3], partial_reset_step=19, max_action_abs_error=max(errors))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


@torch.no_grad()
def validate_recording(module, recording, checkpoint_hash, device='cpu'):
    import numpy as np
    meta = json.loads((recording.parent/'metadata.json').read_text())
    if meta['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Recording belongs to a different checkpoint')
    with np.load(recording, allow_pickle=False) as data:
        obs = torch.from_numpy(data['obs_t'].copy())
        history = torch.from_numpy(data['history_t'].copy())
        current = torch.from_numpy(data['film_condition_t'].copy()).reshape(-1,5,6)
        expected = torch.from_numpy(data['action_raw'].copy())
        steps, trials = data['step'].copy(), data['trial_id'].copy()
    batch = int(meta['resolved_environment']['env']['num_envs'])
    usable = len(obs)//batch*batch
    if usable == 0:
        raise ValueError('Recording has no complete inference batch')
    for i in range(0,usable,batch):
        if len(set(steps[i:i+batch])) != 1 or len(set(trials[i:i+batch])) != batch:
            raise ValueError('Recording does not preserve full inference batch ordering')
    obs, history, current, expected = [x[:usable].to(device) for x in (obs,history,current,expected)]
    module = module.to(device)
    # Saved conditioning includes the current error. Reconstruct previous FIFO;
    # its discarded oldest frame has no influence on this action.
    previous = torch.cat((torch.zeros_like(current[:, :1]),current[:, :4]),1)
    outputs, errors = [], []
    for i in range(0,len(obs),batch):
        action, updated = module(obs[i:i+batch], history[i:i+batch], previous[i:i+batch])
        outputs.append(action)
        errors.append(updated)
    predicted = torch.cat(outputs)
    torch.testing.assert_close(torch.cat(errors), current, rtol=2e-4, atol=2e-4)
    # Recording GPU versus export CPU can differ in reduction precision.
    torch.testing.assert_close(predicted, expected, rtol=2e-4, atol=2e-4)
    module.cpu()
    return dict(path=str(recording), samples=len(obs), skipped_incomplete_batch_rows=len(steps)-usable,
                device=device, inference_batch_size=batch, precision=torch.get_float32_matmul_precision(),
                max_action_abs_error=float((predicted-expected).abs().max()),
                mean_action_abs_error=float((predicted-expected).abs().mean()), atol=2e-4, rtol=2e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Saved training config, never current task defaults')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--recording', type=Path, help='Optional transitions NPZ from this exact checkpoint for saved-action parity')
    parser.add_argument('--recording-device', default='cpu')
    parser.add_argument('--recording-precision', choices=['highest','high','medium'], default='highest')
    args = parser.parse_args()
    config = args.config or args.checkpoint.parent/'b1z1_pact_config.py'
    torch.set_num_threads(1)
    actor, model, metadata = build(args.checkpoint, config)
    scripted = torch.jit.script(model)
    checks = validate(actor, scripted)
    checkpoint_hash = sha(args.checkpoint)
    if args.recording:
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision(args.recording_precision)
            checks['recorded_actions'] = validate_recording(scripted,args.recording,checkpoint_hash,args.recording_device)
        finally:
            scripted.cpu()
            torch.set_float32_matmul_precision(precision)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'policy.pt'
    metadata.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                    config_sha256=sha(config), torch_version=torch.__version__)
    torch.jit.save(scripted, str(path), _extra_files={'metadata.json':json.dumps(metadata)})
    checks['reloaded'] = validate(actor, torch.jit.load(str(path)))
    metadata.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        config_sha256=sha(config), artifact_sha256=sha(path), torch_version=torch.__version__,
        worktree_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        exporter_sha256=sha(Path(__file__)), deployment_source_sha256=sha(ROOT/'rsl_rl/b1z1_legacy_deployment.py'),
        validation=checks)
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(ROOT/'docs/b1z1_legacy_pact_jit.md', args.output_dir/'README.md')
    print(json.dumps(dict(artifact=str(path.resolve()), validation=checks),indent=2))


if __name__ == '__main__':
    main()
