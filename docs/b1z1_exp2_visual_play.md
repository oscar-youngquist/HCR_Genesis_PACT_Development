# Visual inspection of Experiment 2

Activate the Isaac Lab training environment, then run from the repository root:

```bash
bash legged_gym/scripts/play_b1z1_exp2.sh \
  --approach pact --profile pushing \
  --run /path/to/run --ckpt 70000 --gpu cuda:0
```

`--approach` accepts `pact`/`pilot`, `pos`, `unifp`, `unifp-reject`, or any
registered B1Z1 task/ablation name. `--task` is an equivalent flag. `--profile`
selects `pushing`, `pulling`, or `carrying`. `--ckpt -1` selects the latest
checkpoint using the existing evaluator resolver.

Exactly one robot is constructed and rendered. The existing PACT play
visualizer shows the nominal EE target in yellow, measured EE in blue, and
the applied world-frame EE force as a red arrow (default 0.005 m/N). The
camera follows the torso; `--no-follow-robot` fixes it after each trial starts.

Defaults are one loaded trial, evaluation seed 0, the current Exp2 finish time,
real-time pacing, and cuda:0. Useful options:

- `--loading unloaded`: same motions without imposed forces.
- `--episodes 3 --trial-start 0`: inspect three sequential deterministic trials.
- `--speed 0.5`: half-speed pacing; simulation timestep and schedules are unchanged.
- `--headless --max-steps 4`: tiny startup/control smoke check without rendering.

Playback reads the **current** `EVAL_CONFIG` and `Scenario` implementation in
`eval_b1z1_exp2.py`/`b1z1_experiment2.py`; it does not duplicate task timings,
load levels, or EE trajectories. Loads follow the initial heading/world-z
conventions from Exp2, with zero torso loading. Policy inference precedes
interval-start force application; nominal successor targets are refreshed
through the same evaluation hooks. Trials end on the same physical failures
or their horizon. Stage transitions and failure reasons print to the terminal.

Native architectures, gains, scaling and deterministic inference are preserved.
Original UniFP uses the evaluator's prediction-only rejection interface.
No training runner, optimizer, BARD teacher, or dataset writer is constructed.
No transition files or metrics are collected. This is visual playback, not a
replacement for the Experiment 2 data-collection launcher.
