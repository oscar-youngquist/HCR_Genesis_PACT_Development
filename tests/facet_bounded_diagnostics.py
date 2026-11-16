"""Short post-training diagnostics; test-local actions/commands, no performance claim."""
import time
import torch


@torch.no_grad()
def evaluate(runner, steps=128):
    env = runner.env
    output = {}
    policy = runner.get_inference_policy()
    # Longer diagnostic episodes expose a sustained EE plateau after the short
    # base pulse. These changes are local to this smoke process, after training.
    env.max_episode_length = int(2./env.dt)
    env.cfg.facet_forces.cycle_seconds = 2.
    env.cfg.facet_forces.ee_duration = 1.2
    for stochastic in (False, True):
        env.reset_idx(env.all_env_ids)
        hidden = torch.zeros_like(runner.hidden)
        samples = []
        started = time.perf_counter()
        for _ in range(steps):
            # Standing and forward walking represented together; no new sampler.
            env.commands[:, :3] = 0
            env.commands[env.num_envs//2:, 0] = .4
            # Refresh the command block without advancing sensor histories.
            from legged_gym.envs.b1z1.b1z1_facet.observations import nominal_command
            env._facet_observations.packet['observation'][:, :9] = nominal_command(env)
            packet = env.get_facet_observations()
            action, hidden = policy(packet['observation'], hidden, packet['is_init'])
            if stochastic:
                action = action+torch.randn_like(action)*runner.model.student.std.clamp_min(1e-6)
            result = env.step(action)
            snapshot = env.get_facet_tracking()
            ref = env.get_facet_references()
            # Per-step nominal errors are successor diagnostics; terminal failures
            # are counted even where reset has made successor errors unavailable.
            valid = ~result[5].bool()
            ee = (env.simulator.ee_pos-env.curr_ee_goal_cart_world).norm(dim=-1)
            base = (env.simulator.base_lin_vel[:, :2]-env.commands[:, :2]).norm(dim=-1)
            normal = torch.where(valid, ee, torch.nan)
            torque = getattr(env.simulator, 'executed_torques', None)
            limits = getattr(env.simulator, 'torque_limits', None)
            saturation = ((torque.abs() >= 1.1*limits-1e-5).float().mean(-1)
                          if torque is not None and limits is not None else ee*torch.nan)
            failed = torch.zeros_like(valid)
            terminal = env.get_facet_terminal_observations()
            if terminal is not None:
                failed[terminal['ids']] = terminal['failures']
            samples.append(torch.stack((normal, torch.where(valid, base, torch.nan),
                result[5].float(), failed.float(), ref['applied_ee_force_world'].norm(dim=-1),
                ref['applied_base_force_world'].norm(dim=-1), ref['ee_offset_task'].norm(dim=-1),
                ref['ee_relative_speed'], ref['boundary_active'].float(), saturation,
                torch.where(snapshot['valid'], snapshot['errors_squared'][..., 4], torch.nan).nanmean(-1)), -1))
        data = torch.stack(samples).cpu()
        names = ('nominal_ee_error_m', 'nominal_planar_error_mps', 'episode_end', 'failure',
                 'ee_load_N', 'base_load_N', 'offset_m', 'relative_speed_mps',
                 'boundary_fraction', 'torque_saturation_fraction', 'reference_ee_mse')
        modes = {}
        for label, rows in (('standing', slice(0, env.num_envs//2)), ('walking', slice(env.num_envs//2, None))):
            d = data[:, rows]
            modes[label] = {name: dict(mean=float(d[..., i].nanmean()) if d[..., i].isfinite().any() else None,
                valid_count=int(d[..., i].isfinite().sum())) for i, name in enumerate(names)}
            modes[label]['failures'] = int(d[..., 3].sum())
            for i, name in enumerate(names):
                finite = d[..., i][d[..., i].isfinite()]
                if finite.numel():
                    modes[label][name].update(median=float(finite.median()),
                        p95=float(finite.quantile(.95)), maximum=float(finite.max()))
        output['stochastic' if stochastic else 'deterministic'] = dict(modes=modes,
            ee_peak_N=float(data[..., 4].max()), base_peak_N=float(data[..., 5].max()),
            wall_seconds=time.perf_counter()-started, environment_transitions=steps*env.num_envs)
    output['protocol'] = dict(control_dt=env.dt, episode_seconds=2., ee_duration_seconds=1.2,
        load_onset_seconds=env.cfg.facet_forces.start_seconds,
        interpretation='Short integration smoke, not trained-policy performance; modes use separate reset samples.')
    return output
