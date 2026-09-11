"""Resolve effective QP solvers before importing a simulator or activating conda."""
import ast
from pathlib import Path


def effective_solvers(arguments):
    options = {}
    for i, argument in enumerate(arguments):
        key, equal, value = argument.partition('=')
        if key in ('--task','--qp_solver','--rollout_qp_solver','--ppo_qp_solver'):
            if equal or i+1 < len(arguments):
                options[key] = value if equal else arguments[i+1]
    task = options.get('--task','go2_hard_pact_full_isaaclab')
    if not any(k in options for k in ('--qp_solver','--rollout_qp_solver','--ppo_qp_solver')):
        if any(task.startswith('go2_hard_pact_'+variant+'_') for variant in
               ('pos','baseline','soft','soft_penalty','inverse','rollout')):
            return ()  # These tasks do not execute/replay QPs by default.
    config = Path(__file__).resolve().parents[1] / 'envs/go2/go2_hard_pact/go2_hard_pact_config.py'
    defaults = None
    # Read the literal configuration without importing legged_gym/Isaac Sim.
    for node in ast.walk(ast.parse(config.read_text())):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='hard_pact_qp' for t in node.targets):
            defaults = {ast.literal_eval(k):ast.literal_eval(v) for k,v in zip(node.value.keys,node.value.values)
                        if isinstance(k,ast.Constant) and k.value in ('qp_solver','rollout_qp_solver','ppo_qp_solver')}
    if defaults is None:
        raise RuntimeError('Cannot resolve HardPACT QP configuration before simulator startup')
    base = options.get('--qp_solver',defaults['qp_solver'])
    return tuple(options.get('--'+phase+'_qp_solver') or defaults.get(phase+'_qp_solver') or base
                 for phase in ('rollout','ppo'))


if __name__ == '__main__':
    import sys
    print('\n'.join(effective_solvers(sys.argv[1:])))
