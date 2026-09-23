# B1Z1 position-only PPO

Task: `b1z1_ppo_pos`. Use an existing **coupled PACT simulator backend**, such as
`SIMULATOR=isaaclab_b1z1_pact`, not a PACT-Pos backend. The environment accepts
one position action per configured learned joint and appends constant zeros
only at the simulator interface. This executes PACT's existing weighted PD
branch, including its delay, gains, clipping, passive-joint handling, and
backend-specific torque limits. No feedforward torque is learned or applied.

`B1Z1PPOPosCfg` and `B1Z1PPOPosCfgPPO` inherit PACT settings. The new standalone
`ActorCriticB1Z1PPOPos(nn.Module)` uses the same MLP builder, hidden widths,
activations, and default linear initialization. Its actor sees only the current
policy observation; its critic sees the unchanged privileged stack. Gaussian
noise settings use the position half of an inherited coupled-action profile,
or the inherited scalar. No context, FiLM, decoder, or torque-head parameters
are constructed.

The existing two action-history blocks remain in the observation schema:
position actions and executed PD torques divided by the existing per-joint
torque-action scale. The latter is a controller command, not privileged measured
contact force. Environment history compatibility retains one current frame;
PPO storage holds current observations and critic inputs only.

Physical disturbances and all existing reward scales/curricula remain inherited.
Only this task disables the force-induced EE-target displacement. All target
consumers, including observations, rewards, and debug views, use the nominal
scheduled target. No force-command adaptation or estimator cancellation runs.

Standard PPO supplies rollout storage, GAE, clipped objectives, adaptive learning
rate, and gradient clipping. The adapter retains PACT's AdamW weight-decay and
adaptive-entropy conventions, environment/reward/force/domain-randomization
curricula, and checkpointed curriculum progress. Checkpoints are tagged for the
new architecture; existing checkpoints are not modified or interchangeable.

Example inside the existing IsaacLab environment (physical GPU 1):

```bash
SIMULATOR=isaaclab_b1z1_pact python legged_gym/scripts/train.py \
    --task=b1z1_ppo_pos --headless --gpu=cuda:1
```

Focused tests: `tests/test_b1z1_ppo_pos.py` (CPU; simulator-independent).
