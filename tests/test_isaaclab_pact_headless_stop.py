"""Headless training and evaluation must not enter the interactive STOP loop."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import legged_gym.envs  # Use the training entrypoint's registration/import order.
from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator
from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT


@pytest.mark.parametrize("headless,subscribed", [(True, True), (False, True), (True, False)])
def test_pact_constructor_removes_only_headless_stop_subscription(monkeypatch, headless, subscribed):
    handle = Mock() if subscribed else None
    context = SimpleNamespace(_app_control_on_stop_handle=handle)
    config = SimpleNamespace(env=SimpleNamespace(episode_length_s=20.))

    def initialize(sim, cfg, params, device, is_headless):
        sim._sim, sim._cfg, sim._headless = context, cfg, is_headless

    monkeypatch.setattr(IsaacLabSimulator, "__init__", initialize)
    sim = IsaacLabSimulator_PACT(config, {"dt": .005}, "cpu", headless)
    assert sim._sim is context  # Do not clear the live simulator singleton.
    assert config.env.episode_length_s == 20.  # Training timeouts are intentional.
    if headless:
        assert context._app_control_on_stop_handle is None
        if handle is not None:
            handle.unsubscribe.assert_called_once_with()
    else:
        assert context._app_control_on_stop_handle is handle
        handle.unsubscribe.assert_not_called()
