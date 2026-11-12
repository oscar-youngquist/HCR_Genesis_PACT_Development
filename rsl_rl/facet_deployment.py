"""Portable FACET student: torch only, no environment/runner/privileged inputs.

Sensor histories are explicit state. Caller supplies noisy (if desired) sensors
in SI units at control boundaries; no randomness or reference dynamics lives here.
"""
from copy import deepcopy
import json
from pathlib import Path
import torch
from torch import nn
from rsl_rl.facet_model import FACETModel
from rsl_rl.facet_contract import checkpoint_contract


class StudentMean(nn.Module):
    def __init__(self, model, ema, actor):
        super().__init__()
        self.normalizer = deepcopy(model.obs_norm)
        self.features = deepcopy(ema.mlp)
        self.gru = deepcopy(ema.gru)
        self.recurrent_norm = deepcopy(ema.norm)
        # Reconstruction outputs are not needed for deployment.
        self.latent = nn.Linear(128, 160)
        with torch.no_grad():
            self.latent.weight.copy_(ema.out.weight[:160])
            self.latent.bias.copy_(ema.out.bias[:160])
        self.actor_mlp, self.actor_mean = deepcopy(actor.mlp), deepcopy(actor.mean)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor, is_init: torch.Tensor):
        obs = self.normalizer(observation)
        features = self.features(obs)
        h = self.gru(features, hidden*(~is_init.to(torch.bool)).reshape(-1, 1))
        latent = self.latent(features+self.recurrent_norm(h))
        return self.actor_mean(self.actor_mlp(torch.cat((obs, latent), -1))), h


class SensorStudent(nn.Module):
    """Mean action plus explicit next histories/hidden state; no execution clipping.

    q/qd arrive in recorded physical joint order (19). issued_action is the
    PREVIOUS issued position action; history clips it before execution delay.
    nominal = [vx,vy,yaw_rate, EE mixed XYZ, EE trajectory mixed XYZ velocity].
    impedance uses the ten recorded parameter_names in their exact order.
    """
    def __init__(self, student, contract):
        super().__init__()
        self.student = student
        names, observed = contract['joints'], contract['schema']['observed_joint_names']
        self.register_buffer('joint_ids', torch.tensor([names.index(n) for n in observed]))
        arm = ('z1_waist', 'z1_shoulder', 'z1_elbow', 'z1_wrist_angle', 'z1_forearm_roll', 'z1_wrist_rotate')
        self.register_buffer('arm_ids', torch.tensor([observed.index(n) for n in arm]))
        for name in ('joint_offsets', 'link00_offset', 'ee_offset'):
            self.register_buffer(name, torch.tensor(contract['kinematics'][name], dtype=torch.float32))
        self.joints = len(observed)
        self.clip_actions = float(contract['clip_actions'])

    def fk(self, q: torch.Tensor):
        # Same six-axis chain as z1_arm_kinematics; kept torch-only so importing
        # the portable builder never initializes legged_gym/simulator selection.
        rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(q.shape[0], 3, 3).clone()
        position = self.link00_offset.expand(q.shape[0], 3).clone()
        for i in range(6):
            position = position+torch.einsum('nij,j->ni', rotation, self.joint_offsets[i])
            c, s = q[:, i].cos()[:, None], q[:, i].sin()[:, None]
            updated = rotation.clone()
            if i == 0 or i == 4:
                updated[:, :, 0] = c*rotation[:, :, 0]+s*rotation[:, :, 1]
                updated[:, :, 1] = -s*rotation[:, :, 0]+c*rotation[:, :, 1]
            elif i < 4:
                updated[:, :, 0] = c*rotation[:, :, 0]-s*rotation[:, :, 2]
                updated[:, :, 2] = s*rotation[:, :, 0]+c*rotation[:, :, 2]
            else:
                updated[:, :, 1] = c*rotation[:, :, 1]+s*rotation[:, :, 2]
                updated[:, :, 2] = -s*rotation[:, :, 1]+c*rotation[:, :, 2]
            rotation = updated
        return position+torch.einsum('nij,j->ni', rotation, self.ee_offset)

    @torch.jit.export
    def observation(self, nominal: torch.Tensor, impedance: torch.Tensor, gravity: torch.Tensor,
                    q: torch.Tensor, qd: torch.Tensor, issued_action: torch.Tensor,
                    is_init: torch.Tensor, sensors: torch.Tensor, actions: torch.Tensor):
        q, qd = q.index_select(1, self.joint_ids), qd.index_select(1, self.joint_ids)
        current = torch.cat((gravity, q, qd), -1)
        init = is_init.to(torch.bool).reshape(-1, 1, 1)
        sensors = torch.where(init, current[:, None].expand(-1, 4, -1),
                              torch.cat((current[:, None], sensors[:, :3]), 1))
        actions = torch.where(init, torch.zeros_like(actions),
            torch.cat((issued_action.clamp(-self.clip_actions, self.clip_actions)[:, None], actions[:, :2]), 1))
        packet = torch.cat((nominal, impedance, sensors[:, :, :3].flatten(1),
            sensors[:, :, 3:3+self.joints].flatten(1), sensors[:, :, 3+self.joints:].flatten(1),
            actions.flatten(1), self.fk(q.index_select(1, self.arm_ids))), -1)
        return packet, sensors, actions

    def forward(self, nominal: torch.Tensor, impedance: torch.Tensor, gravity: torch.Tensor,
                q: torch.Tensor, qd: torch.Tensor, issued_action: torch.Tensor,
                is_init: torch.Tensor, sensors: torch.Tensor, actions: torch.Tensor, hidden: torch.Tensor):
        obs, sensors, actions = self.observation(nominal, impedance, gravity, q, qd, issued_action, is_init, sensors, actions)
        action, hidden = self.student(obs, hidden, is_init)
        return action, hidden, sensors, actions


def export_student(checkpoint, output, copy_teacher=False, legacy_contract=None):
    """Explicit copied-teacher export is a separate artifact, never checkpoint mutation."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    contract = checkpoint_contract(state, checkpoint, legacy_contract)
    if contract.get('kinematics') is None:
        raise ValueError('Export requires verified FK geometry in metadata; supply explicit legacy_contract')
    alg = state['algorithm']
    initialized = alg.get('student_initialized', alg['transitioned'])
    if alg['phase'] not in ('teacher', 'adaptation', 'student') or initialized != alg['transitioned'] or initialized != (alg['phase'] != 'teacher'):
        raise ValueError('Inconsistent FACET checkpoint phase/initialization status')
    if not initialized and not copy_teacher:
        raise ValueError('Uninitialized student: explicitly select --copy-teacher for estimated-feature evaluation')
    if initialized and copy_teacher:
        raise ValueError('--copy-teacher is only for uninitialized teacher checkpoints')
    dims = [sum(int(width) for _, width in contract['schema'][name]) for name in ('observation_blocks', 'privileged_blocks')]
    model = FACETModel(*dims)
    model.load_state_dict(alg['model'], strict=True)
    ema = deepcopy(model.estimator)
    ema.load_state_dict(alg['ema'], strict=True)
    student = StudentMean(model, ema, model.student if initialized else model.teacher)
    artifact = SensorStudent(student, contract).eval().requires_grad_(False)
    metadata = dict(format='facet_sensor_student_torchscript_v1', contract=contract,
        source_checkpoint=str(Path(checkpoint).resolve()), source_iteration=state['iteration'],
        source_phase=alg['phase'], copied_teacher=not initialized,
        phase_interactions=state.get('phase_interactions'), accounting_exact=state.get('accounting_exact', False),
        training_seed=state['config'].get('seed'), phase_schedule=state['config']['algorithm'],
        force_curriculum=state.get('force_curriculum'), domain_curriculum=state.get('domain_curriculum'))
    import hashlib
    with open(checkpoint, 'rb') as stream:
        metadata['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    # Curriculum EMA uses -inf before its first observation. It is unavailable
    # informational metadata here, not a deployment value or a JSON number.
    metadata = json.loads(json.dumps(metadata), parse_constant=lambda _: None)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(artifact).save(str(output), _extra_files={'metadata.json': json.dumps(metadata, allow_nan=False)})
    Path(str(output)+'.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return metadata


def load_student(path, device='cpu'):
    extra = {'metadata.json': ''}
    module = torch.jit.load(str(path), map_location=device, _extra_files=extra).eval()
    metadata = json.loads(extra['metadata.json'])
    if metadata['format'] != 'facet_sensor_student_torchscript_v1':
        raise ValueError('Unknown FACET deployment format')
    return module, metadata


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--copy-teacher', action='store_true')
    parser.add_argument('--legacy-contract', help='Explicit verified metadata JSON for older checkpoints')
    args = parser.parse_args()
    contract = json.loads(Path(args.legacy_contract).read_text()) if args.legacy_contract else None
    print(json.dumps(export_student(args.checkpoint, args.output, args.copy_teacher, contract), indent=2))
