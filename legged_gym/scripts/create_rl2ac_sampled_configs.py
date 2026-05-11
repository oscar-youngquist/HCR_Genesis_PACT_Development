import os
import numpy as np


def latin_hypercube_sample(param_ranges, n_samples, seed=None):
    """
    Simple standalone Latin-hypercube sampler.

    Args:
        param_ranges: dict mapping parameter names to (min, max)
        n_samples: number of sampled configurations
        seed: optional random seed

    Returns:
        list[dict]: one sampled config per entry
    """
    rng = np.random.default_rng(seed)

    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    lhs_unit = np.zeros((n_samples, n_params))

    for j in range(n_params):
        bin_edges = np.linspace(0.0, 1.0, n_samples + 1)
        lower = bin_edges[:-1]
        upper = bin_edges[1:]

        points = rng.uniform(lower, upper)
        rng.shuffle(points)

        lhs_unit[:, j] = points

    samples = []
    for i in range(n_samples):
        cfg = {}
        for j, name in enumerate(param_names):
            lo, hi = param_ranges[name]
            cfg[name] = lo + lhs_unit[i, j] * (hi - lo)
        samples.append(cfg)

    return samples


def format_float(x, precision=5):
    """
    Format floats compactly for shell commands.
    """
    return f"{x:.{precision}g}"


def make_rl2ac_command(
    cfg,
    task="go1_rl2ac",
    gpu="cuda:0",
    seed=1,
    num_eps=2.00,
    num_envs=50,
    terrain_type="rough",
    disturbance_type="payload",
    log_path="exp_data_corl_rl2ac/sample_param_search",
    script_name="play_exp_rl2ac_tune.py",
):
    """
    Construct one RL2AC tuning command.
    """
    return (
        f"python {script_name} "
        f"--task={task} "
        f"--gpu={gpu} "
        f"--headless "
        f"--seed={seed} "
        f"--num_eps={num_eps:.2f} "
        f"--num_envs={num_envs} "
        f"--terrain_type={terrain_type} "
        f"--disturbance_type={disturbance_type} "
        f"--log "
        f"--log_path={log_path} "
        f"--alpha={format_float(cfg['alpha'])} "
        f"--kappa={format_float(cfg['kappa'])} "
        f"--lambda_0={format_float(cfg['lambda_0'])} "
        f"--k_0={format_float(cfg['k_0'])}"
    )


def write_lhs_sweep_script(
    output_script_path="run_rl2ac_lhs_sweep.sh",
    n_samples=100,
    seed=0,
):
    """
    Generate a bash script containing RL2AC parameter tuning commands.
    """

    param_ranges = {
        "alpha": (20.0, 55.0),
        "kappa": (0.02, 0.25),
        "lambda_0": (0.1, 1.5),
        "k_0": (2.0, 20.0),
    }

    samples = latin_hypercube_sample(
        param_ranges=param_ranges,
        n_samples=n_samples,
        seed=seed,
    )

    lines = []

    lines.append("#!/bin/bash")
    lines.append("")
    lines.append(". /home/oyoungquist/anaconda3/etc/profile.d/conda.sh")
    lines.append("")
    lines.append("conda activate /home/oyoungquist/.conda/envs/genesis_lr")
    lines.append("")
    lines.append("export SIMULATOR=genesis_pact_rl2ac")
    lines.append("")

    # Reference configurations from your existing script format
    lines.append("# Sanity checks w/ paper parameters")
    paper_cfg = {
        "alpha": 50.0,
        "kappa": 1.2,
        "lambda_0": 3.0,
        "k_0": 20.0,
    }
    lines.append(make_rl2ac_command(paper_cfg))
    lines.append("")

    lines.append("# Conservative benchmark")
    conservative_cfg = {
        "alpha": 10.0,
        "kappa": 0.1,
        "lambda_0": 0.3,
        "k_0": 5.0,
    }
    lines.append(make_rl2ac_command(conservative_cfg))
    lines.append("")

    lines.append("# Best zero-failure result from initial coarse grid sweep")
    best_zero_failure_cfg = {
        "alpha": 35.0,
        "kappa": 0.1,
        "lambda_0": 0.3,
        "k_0": 5.0,
    }
    lines.append(make_rl2ac_command(best_zero_failure_cfg))
    lines.append("")

    lines.append("# Best low-kappa linear tracking result from initial coarse grid sweep")
    best_low_kappa_tracking_cfg = {
        "alpha": 50.0,
        "kappa": 0.1,
        "lambda_0": 0.3,
        "k_0": 5.0,
    }
    lines.append(make_rl2ac_command(best_low_kappa_tracking_cfg))
    lines.append("")

    lines.append(f"# Latin-hypercube sweep: n_samples={n_samples}, seed={seed}")
    for i, cfg in enumerate(samples):
        lines.append(f"# LHS sample {i:03d}")
        lines.append(make_rl2ac_command(cfg))
        lines.append("")

    with open(output_script_path, "w") as f:
        f.write("\n".join(lines))

    os.chmod(output_script_path, 0o755)

    print(f"Wrote sweep script to: {output_script_path}")
    print(f"Generated {n_samples} LHS samples + 2 reference runs")


if __name__ == "__main__":
    write_lhs_sweep_script(
        output_script_path="run_rl2ac_lhs_sweep.sh",
        n_samples=100,
        seed=1,
    )