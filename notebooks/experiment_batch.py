"""Plan independent per-task summaries from completed recording metadata."""
from copy import deepcopy
from pathlib import Path

from exp1_analysis import discover, read_json


def plan_summaries(config, experiment):
    """Keep validation/statistics in existing summarizers; never merge methods."""
    single = config.get("task")
    selected = config.get("tasks")
    if single and selected is not None:
        raise ValueError("Use task (legacy single output) OR tasks (batch), not both")
    if selected is not None and (not isinstance(selected, list) or not selected
                                or any(not isinstance(t, str) or not t for t in selected)
                                or len(set(selected)) != len(selected)):
        raise ValueError("tasks must be None or a nonempty list of unique task names")
    options = config.get("task_options", {})
    runs = {}
    for path in discover(config["input_roots"], "metadata.json"):
        if not (path.parent / "COMPLETE").is_file():
            continue
        meta = read_json(path)
        # Older Exp1 recordings predate the explicit experiment field.
        if meta.get("experiment", 1) != experiment or not meta.get("task"):
            continue
        if experiment == 2:
            loading = config.get("loading", ["loaded"])
            loading = [loading] if isinstance(loading, str) else loading
            if meta.get("arguments", {}).get("loading") not in loading:
                continue
        runs.setdefault(meta["task"], []).append((path, meta))
    tasks = [single] if single else selected if selected is not None else sorted(runs)
    if not tasks:
        raise ValueError(f"No completed Experiment {experiment} recordings found")
    missing = set(tasks) - runs.keys()
    if missing:
        raise ValueError(f"No completed recordings for requested tasks: {sorted(missing)}")
    plans, inventory = [], []
    for task in tasks:
        # Registered task names become directory names; reject unsafe custom metadata.
        if task in (".", "..") or Path(task).name != task or "\\" in task:
            raise ValueError(f"Unsafe task name: {task!r}")
        overrides = options.get(task, {})
        if set(overrides) - {"display_name", "checkpoint_sha256"}:
            raise ValueError(f"Unsupported task_options for {task}: {sorted(overrides)}")
        settings = deepcopy(config)
        settings.update(task=task, display_name=(config.get("display_name") if single else None)
                        or config.get("display_names", {}).get(task, task))
        settings.update(overrides)
        matching = [(p, m) for p, m in runs[task] if not settings.get("checkpoint_sha256")
                    or m.get("checkpoint_sha256") == settings["checkpoint_sha256"]]
        if not matching:
            raise ValueError(f"No completed recordings for {task}'s checkpoint selection")
        # Pass exact metadata paths so mixed experiment roots cannot leak into Exp1.
        settings["input_roots"] = [str(p) for p, _ in matching]
        settings["output_dir"] = str(Path(config["output_dir"]) / task) if not single else config["output_dir"]
        plans.append(settings)
        inventory.append(dict(task=task, runs=len(matching),
            evaluation_seeds=sorted({m["evaluation_seed"] for _, m in matching}),
            checkpoints=sorted({m["checkpoint_sha256"] for _, m in matching}),
            output_dir=settings["output_dir"]))
    return plans, inventory
