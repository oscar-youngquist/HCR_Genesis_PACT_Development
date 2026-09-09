"""Replay a trusted local QP exception capture without starting a simulator."""
import argparse
import torch
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warp-path", help="Directory containing the captured Warp package; reproduce simulator import order")
    args = parser.parse_args()
    if args.warp_path:
        sys.path.insert(0, args.warp_path)
        import warp
        print(f"Replay Warp: {warp.__version__} ({warp.__file__})")
    from rsl_rl.algorithms.hard_pact_qp_capture import replay_snapshot
    payload = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    print({k: payload[k] for k in ("solver", "differentiable", "relaxed_contact", "elastic", "exception", "packages")})
    solution = replay_snapshot(payload, args.device)
    if payload["differentiable"]:
        solution.square().mean().backward()
    print(f"Replay completed: shape={tuple(solution.shape)}, finite={bool(torch.isfinite(solution).all())}")
