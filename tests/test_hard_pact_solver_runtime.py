"""HardPACT-only Isaac Lab/cuPIQP import ordering and legacy isolation."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / 'legged_gym/scripts/train_hard_pact.py'
spec = importlib.util.spec_from_file_location('hard_pact_train_entry', path)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.mark.parametrize('option', ['--qp_solver', '--rollout_qp_solver', '--ppo_qp_solver'])
@pytest.mark.parametrize('equals', [False, True])
def test_cupiqp_pins_compatible_runtime_and_preserves_old_api(monkeypatch, option, equals):
    array, indexed = object(), object()
    warp = SimpleNamespace(__version__='1.17.0', __file__='/env/warp/__init__.py',
                           types=SimpleNamespace(), array=array, indexedarray=indexed)
    monkeypatch.setenv('SIMULATOR', 'isaaclab')
    monkeypatch.setitem(sys.modules, 'warp', warp)
    entry.prepare_solver_runtime([f'{option}=cupiqp'] if equals else [option, 'cupiqp'])
    assert warp.types.array is array
    assert warp.types.indexedarray is indexed


@pytest.mark.parametrize('simulator,solver', [('isaaclab', 'qpth'), ('genesis_pact', 'cupiqp')])
def test_unaffected_runtime_is_not_touched(monkeypatch, simulator, solver):
    # Any attempt to access this sentinel as Warp would fail immediately.
    monkeypatch.setenv('SIMULATOR', simulator)
    monkeypatch.setitem(sys.modules, 'warp', object())
    entry.prepare_solver_runtime(['--qp_solver', solver])


def test_old_warp_fails_before_simulator_startup(monkeypatch):
    monkeypatch.setenv('SIMULATOR', 'isaaclab')
    monkeypatch.setitem(sys.modules, 'warp', SimpleNamespace(__version__='1.8.2', __file__='/bundled/warp'))
    with pytest.raises(RuntimeError, match='requires Warp >=1.12'):
        entry.prepare_solver_runtime(['--qp_solver', 'cupiqp'])
