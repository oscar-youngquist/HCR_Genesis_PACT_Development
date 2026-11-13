"""HardPACT/Pos adapter for the shared envelope; legacy PACT is untouched."""
from copy import deepcopy
from pathlib import Path
import torch
from . import training_checkpoint as tc


def save(runner, checkpoint, path, *, stage_end=False):
    checkpoint[tc.KEY] = tc.envelope(runner, stage_end=stage_end)
    checkpoint['iter'] = checkpoint[tc.KEY]['next_iteration']
    tc.atomic_save(checkpoint, path, getattr(runner, '_checkpoint_provenance', None))


def load(runner, path, *, operation='resume', learning_rate=None, allow_legacy=False,
         target_stage=None, stage_local_initial=0.):
    tc.boundary(runner)
    if operation not in tc.OPERATIONS: raise ValueError('Unknown checkpoint operation')
    if operation == 'weights_only' and learning_rate is not None:
        raise ValueError('weights_only does not restore optimizers or apply fine-tuning LR')
    if target_stage is not None and operation != 'stage_transfer': raise ValueError('Stage changes require stage_transfer')
    if operation == 'stage_transfer' and not target_stage: raise ValueError('Stage transfer requires a target stage')
    if not 0 <= stage_local_initial <= 1: raise ValueError('Stage-local severity must lie in [0,1]')
    if runner.log_dir and Path(runner.log_dir).resolve() == Path(path).resolve().parent:
        raise ValueError('Use a destination log directory, never the source run')
    data = torch.load(path, map_location=runner.device, weights_only=False)
    state = data.get(tc.KEY)
    if state is None:
        if operation == 'stage_transfer' or (operation != 'weights_only' and not allow_legacy):
            raise ValueError('Legacy checkpoint lacks a complete training contract; use weights_only or --allow-legacy-checkpoint')
        # Strict key/shape preflight across BOTH modules before either changes.
        for name, key in (('actor_critic', 'model_state_dict'), ('decoder', 'decoder_state_dict')):
            own = getattr(runner.alg, name).state_dict(); old = data[key]
            if own.keys() != old.keys() or any(own[k].shape != old[k].shape for k in own):
                raise ValueError('Legacy weights incompatible: '+name)
        missing = tc.warn_legacy()
        import warnings
        warnings.warn('Legacy HardPACT: loading ONLY networks; optimizer moments and global progress '
                      'are not restored without a named-ownership migration.', RuntimeWarning)
        runner.alg.actor_critic.load_state_dict(data['model_state_dict'], strict=True)
        runner.alg.decoder.load_state_dict(data['decoder_state_dict'], strict=True)
        # Do not guess unnamed optimizer ownership. Explicit legacy initialization
        # is weights-only and advertises that no continuation was performed.
        operation = 'weights_only'
    else:
        tc.validate(runner, data, operation, learning_rate=learning_rate)
        missing = state['unavailable']
        if operation == 'weights_only':
            for name, model in tc.modules(runner).items(): model.load_state_dict(state['models'][name], strict=True)
        else:
            tc.restore_learning(runner, state, learning_rate=learning_rate)
            runner._training_progress_restored = True
            progress = state['transferable_progress']
            runner.env.import_training_progress(progress, transfer=operation == 'stage_transfer')
            qp_state = state['qp_curriculum']
            if qp_state is not None:
                from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum
                runner.qp_curriculum = QPCurriculum(runner.alg.qp_config)
                runner.qp_curriculum.load_state_dict(tc.clear_gate_evidence(qp_state)
                    if operation == 'stage_transfer' else qp_state)
            if operation == 'stage_transfer':
                runner.training_stage = dict(name=target_stage, origin_iteration=state['next_iteration'],
                                             local_progress=stage_local_initial)
                runner.env.initialize_transferred_payload(deepcopy(runner.training_stage))
            tc.restore_rng(runner, state['rng'], transfer=operation == 'stage_transfer')
    runner.alg.storage.clear()
    # clear() normally RETAINS action-delay boundary frames for the next
    # rollout. A fresh episode must explicitly discard those frames as well.
    for name, value in vars(runner.alg.storage).items():
        if torch.is_tensor(value): value.zero_()
        elif name == 'hard_pact_fields' and isinstance(value, dict):
            for tensor in value.values(): tensor.zero_()
    for opt in tc.optimizers(runner).values(): opt.zero_grad(set_to_none=True)
    if hasattr(runner.alg, 'transition'): runner.alg.transition.clear()
    if hasattr(runner.alg, '_clear_rollout_mechanics_cache'): runner.alg._clear_rollout_mechanics_cache()
    for name in tc.wrappers(runner):
        wrapper = getattr(runner.alg, name)
        wrapper.last_objective_grads = wrapper.last_merged_grad = wrapper.last_has_grads = None
    runner.env.reset_training_episodes()
    if hasattr(runner, '_set_hard_pact_qp_iteration'):
        runner._set_hard_pact_qp_iteration(runner.current_learning_iteration)
    runner._checkpoint_provenance = str(Path(path).resolve())
    tc.archive_report(runner, path, state, operation, learning_rate, unavailable=missing)
    return data.get('infos')
