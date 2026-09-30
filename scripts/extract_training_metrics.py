#!/usr/bin/env python3
"""Stream a TensorBoard event file into a compact scalar-only plotting bundle.

Requires tensorboard and numpy (available in the training environments). The
source is read twice, never modified, and never loaded wholly into memory.
Use a completed run or a stable copy of an event file, not a live writer.
"""

import argparse
from collections import Counter
import csv
import fnmatch
import json
import math
from pathlib import Path


# Case-insensitive globs cover UniFP, B1Z1 PACT, and Go2 HardPACT logging.
DEFAULT_PATTERNS = (
    "episode/*", "reward/*", "rewards/*", "tracking/*", "ee/*",
    "train/mean_reward", "train/mean_episode_length", "train/success",
    "*pinn*", "physics/*", "*inverse_residual*", "*rollout_velocity*",
    "*tracking*", "forcecurriculum/*", "domain_rand/*",
    "values/domain_rand*", "loss/*", "adaptation/*",
)


def selected(tag, includes, excludes):
    tag = tag.lower()
    return (any(fnmatch.fnmatchcase(tag, pattern.lower()) for pattern in includes)
            and not any(fnmatch.fnmatchcase(tag, pattern.lower()) for pattern in excludes))


def scalar_values(source):
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.util import tensor_util

    for record in RawEventFileLoader(str(source)).Load():
        event = Event.FromString(record)
        for value in event.summary.value:
            kind = value.WhichOneof("value")
            if kind == "simple_value":
                yield value.tag, event.step, event.wall_time, float(value.simple_value)
            elif kind == "tensor":
                # Exclude text, images, histograms, and non-scalar tensors.
                plugin = value.metadata.plugin_data.plugin_name
                if plugin not in ("", "scalars"):
                    continue
                array = tensor_util.make_ndarray(value.tensor)
                if array.size == 1 and array.dtype.kind in "biuf":
                    yield value.tag, event.step, event.wall_time, float(array.item())


def retained_indices(count, limit):
    """Evenly spaced record indices, retaining endpoints without smoothing."""
    if limit == 0 or count <= limit:
        return range(count)
    return {(i * (count - 1)) // (limit - 1) for i in range(limit)}


def extract(source, output, includes, excludes, limit, list_only=False):
    counts = Counter()
    stats = {}
    stamp = (source.stat().st_size, source.stat().st_mtime_ns)
    print("Scanning scalar tags (pass 1)...", flush=True)
    for tag, step, wall_time, value in scalar_values(source):
        counts[tag] += 1
        if not selected(tag, includes, excludes):
            continue
        if tag not in stats:
            stats[tag] = dict(first_step=step, last_step=step, finite_count=0,
                              nonfinite_count=0, minimum=None, maximum=None)
        item = stats[tag]
        item["last_step"] = step
        if math.isfinite(value):
            item["finite_count"] += 1
            item["minimum"] = value if item["minimum"] is None else min(item["minimum"], value)
            item["maximum"] = value if item["maximum"] is None else max(item["maximum"], value)
        else:
            item["nonfinite_count"] += 1
    for tag in sorted(counts):
        print(f'{"KEEP" if tag in stats else "DROP":4} {counts[tag]:9d}  {tag}')
    if list_only:
        return
    if not stats:
        raise ValueError("No matching scalars; use --list-tags or --include GLOB.")
    if stamp != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError("Source changed during scanning; use a stable copy.")

    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    # Refuse existing outputs so neither the source nor an earlier export is overwritten.
    output.mkdir(parents=True, exist_ok=False)
    keep = {tag: retained_indices(counts[tag], limit) for tag in stats}
    seen, written = Counter(), Counter()
    writer = EventFileWriter(str(output / "tensorboard"))
    print("Writing selected scalars (pass 2)...", flush=True)
    try:
        with (output / "metrics.csv").open("w", newline="") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(("tag", "step", "wall_time", "value"))
            for tag, step, wall_time, value in scalar_values(source):
                if tag not in keep:
                    continue
                index = seen[tag]
                seen[tag] += 1
                if index not in keep[tag]:
                    continue
                csv_writer.writerow((tag, step, wall_time, value))
                writer.add_event(Event(step=step, wall_time=wall_time,
                                       summary=Summary(value=[Summary.Value(
                                           tag=tag, simple_value=value)])))
                written[tag] += 1
    finally:
        writer.close()
    if stamp != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError("Source changed during export; discard this output and use a stable copy.")
    if any(seen[tag] != counts[tag] for tag in stats):
        raise ValueError("Record counts changed between passes; export is incomplete.")
    manifest = dict(
        source=str(source), source_bytes=stamp[0], max_points_per_tag=limit,
        sampling="Evenly spaced record indices per tag; endpoints retained; no averaging.",
        notes="Step and wall_time are original. Duplicate/resumed steps are not merged. "
              "Nonfinite values remain in CSV/events. Min/max use all finite source scalars. "
              "Unsampled transients may not appear in curves.",
        includes=list(includes), excludes=list(excludes),
        kept_tags={tag: dict(stats[tag], source_count=counts[tag], output_count=written[tag])
                   for tag in sorted(stats)},
        dropped_scalar_tags={tag: count for tag, count in sorted(counts.items()) if tag not in stats},
    )
    (output / "summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Kept {sum(written.values()):,} points across {len(written)} metrics.")
    for path in sorted(output.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            print(f"{size / 1024**2:.2f} MiB  {path}")
            if size >= 450 * 1024**2:
                print("WARNING: large output; rerun with fewer --max-points-per-tag or narrower tags.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="One TensorBoard events.out.tfevents.* file")
    parser.add_argument("--output-dir", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--max-points-per-tag", type=int, default=2000,
                        help="Evenly spaced records per metric; 0 keeps all (default: 2000)")
    parser.add_argument("--include", action="append", default=[], help="Additional case-insensitive glob")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude glob; overrides includes")
    parser.add_argument("--no-default-tags", action="store_true", help="Use only --include patterns")
    parser.add_argument("--list-tags", action="store_true", help="List scalar counts/selection without writing")
    args = parser.parse_args()
    if args.max_points_per_tag < 0 or args.max_points_per_tag == 1:
        parser.error("--max-points-per-tag must be 0 or at least 2")
    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Not a file: {source}")
    output = (args.output_dir or source.with_name(source.name + ".summary")).expanduser().resolve()
    if output.exists() and not args.list_tags:
        parser.error(f"Output already exists: {output}")
    includes = (() if args.no_default_tags else DEFAULT_PATTERNS) + tuple(args.include)
    try:
        extract(source, output, includes, args.exclude, args.max_points_per_tag, args.list_tags)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
