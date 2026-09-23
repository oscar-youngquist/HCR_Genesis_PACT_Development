"""Transition-aligned, preallocated CPU replay for coupled B1Z1 PACT."""
import torch


class ReplayBufferB1Z1PACT:
    # Only normalized policy inputs use half precision. Mechanics, targets,
    # torques and rewards must retain their original FP32 precision.
    compact = {"observations", "next_observations", "histories", "next_histories",
               "critic_observations", "next_critic_observations"}

    def __init__(self, capacity=32768, *, n_step=1, device="cpu", pin_memory=True):
        if n_step != 1:
            raise ValueError("B1Z1 FlashSAC replay supports only n_step=1")
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity, self.device = int(capacity), torch.device(device)
        self.pin_memory = pin_memory and self.device.type == "cpu" and torch.cuda.is_available()
        self.data, self.cursor, self.size = {}, 0, 0

    @property
    def estimated_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.data.values())

    def initialize(self, example):
        # Example shapes come from the configured environment, including all
        # optional actor-physics fields. Allocation is outside inference mode.
        with torch.inference_mode(False):
            for name, value in example.items():
                dtype = torch.float16 if name in self.compact else value.dtype
                self.data[name] = torch.empty((self.capacity, *value.shape[1:]),
                    dtype=dtype, device=self.device, pin_memory=self.pin_memory)
        print(f"B1Z1 FlashSAC replay: {self.capacity:,} transitions, "
              f"{self.estimated_bytes / 2**20:.1f} MiB ({self.device}, pinned={self.pin_memory})")

    def add(self, transition):
        if not self.data:
            self.initialize(transition)
        if transition.keys() != self.data.keys():
            raise ValueError("Replay transition schema changed")
        count = len(next(iter(transition.values())))
        if any(len(v) != count or v.shape[1:] != self.data[k].shape[1:] for k, v in transition.items()):
            raise ValueError("Replay fields have inconsistent shapes")
        # Supports vector batches larger than capacity without duplicate scatter indices.
        keep = min(count, self.capacity)
        indices = (torch.arange(keep, device=self.device) + self.cursor + count - keep) % self.capacity
        for name, value in transition.items():
            self.data[name][indices] = value[-keep:].detach().to(self.device, self.data[name].dtype)
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size, device):
        if not self.size:
            raise RuntimeError("Cannot sample empty replay")
        indices = torch.randint(self.size, (batch_size,), device=self.device)
        batch = {k: v[indices].to(device=device, dtype=torch.float32 if v.is_floating_point() else v.dtype,
                                  non_blocking=True) for k, v in self.data.items()}
        # Physics caches are local to this sampled batch, never replay indices.
        batch["indices"] = torch.arange(batch_size, device=device)
        return batch

    def state_dict(self):
        return dict(capacity=self.capacity, cursor=self.cursor, size=self.size,
                    data={k: v[:self.size].clone() for k, v in self.data.items()})

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Checkpoint replay capacity differs from configured capacity")
        if state["data"]:
            self.initialize(state["data"])
            for k, v in state["data"].items():
                self.data[k][:len(v)].copy_(v)
        self.cursor, self.size = state["cursor"], state["size"]
