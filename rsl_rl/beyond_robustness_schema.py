"""Versioned BR packing, independent of simulator and HardPACT observation stacks.

The paper does not enumerate its 62 parameter scalars. Our declared default uses
per-joint Kp/Kd/strength, and per-leg-link realized mass/CoM (12 leg links), plus
base mass/CoM: 88 *meaningful* values, not 62 padded/truncated values. Adapters may
declare another named layout, but must supply every declared field. Physical link
name mapping and simulator property extraction are owned by the payload adapter.
The Stage-3 tray adds one declared realized tray-mass scalar (89 parameters);
the full schema, not just a width/version, is checked on checkpoint loading.
"""
from dataclasses import asdict, dataclass
import math
import torch


@dataclass(frozen=True)
class InputField:
    name: str
    width: int
    units: str

    def __post_init__(self):
        if not self.name or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("Input fields require a name and positive integer width")


PROPRIOCEPTION_FIELDS = (
    InputField("angular_velocity_B", 3, "rad/s"), InputField("gravity_B", 3, "unit vector"),
    InputField("command_B", 3, "m/s,m/s,rad/s"), InputField("joint_position", 12, "rad"),
    InputField("joint_velocity", 12, "rad/s"), InputField("previous_action", 12, "policy units"),
)
LOAD_FIELDS = (InputField("relative_com_position_B", 3, "m"),
               InputField("relative_velocity_B", 3, "m/s"), InputField("mass", 1, "kg"))
PARAMETER_FIELDS = (
    InputField("joint_kp", 12, "Nm/rad"), InputField("joint_kd", 12, "Nm*s/rad"),
    InputField("motor_strength", 12, "ratio"), InputField("leg_link_mass", 12, "kg"),
    InputField("leg_link_com", 36, "m,link frame"), InputField("base_mass", 1, "kg"),
    InputField("base_com", 3, "m,body frame"),
)


@dataclass(frozen=True)
class BeyondRobustnessInputSchema:
    version: int = 3
    actor_clip: float = 100.0
    # Explicit Cartesian sample locations, x-major/y-minor, not a paper-sized buffer.
    terrain_points_x: tuple = tuple(round(i / 10, 1) for i in range(-6, 7))
    terrain_points_y: tuple = tuple(round(i / 10, 1) for i in range(-5, 6))
    parameter_fields: tuple = PARAMETER_FIELDS
    joint_order: tuple = tuple(f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                               for joint in ("hip", "thigh", "calf"))
    foot_order: tuple = ("FR", "FL", "RR", "RL")

    def __post_init__(self):
        if self.version != 3:
            raise ValueError("Incompatible BR observation schema: Stage 4 uses cached sensor noise and clean privileged truth; explicit checkpoint conversion required")
        if not math.isfinite(self.actor_clip) or self.actor_clip <= 0:
            raise ValueError('actor_clip must be finite and positive')
        for name in ("terrain_points_x", "terrain_points_y", "joint_order", "foot_order"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        fields = tuple(InputField(**f) if isinstance(f, dict) else f for f in self.parameter_fields)
        object.__setattr__(self, "parameter_fields", fields)
        if not fields or len({f.name for f in fields}) != len(fields):
            raise ValueError("Parameter fields must be nonempty and uniquely named")
        if (len(self.joint_order) != 12 or len(set(self.joint_order)) != 12 or
                len(self.foot_order) != 4 or len(set(self.foot_order)) != 4):
            raise ValueError("BR requires 12 named joints and four named feet")
        if not all(math.isfinite(x) for x in self.terrain_points_x + self.terrain_points_y):
            raise ValueError("Terrain sample locations must be finite")

    @classmethod
    def from_terrain(cls, terrain, **kwargs):
        def get(key):
            return terrain[key] if isinstance(terrain, dict) else getattr(terrain, key)
        x, y = (get("measured_points_x"), get("measured_points_y")) if get("measure_heights") else ((), ())
        return cls(terrain_points_x=tuple(x), terrain_points_y=tuple(y), **kwargs)

    @property
    def state_fields(self):
        heights = len(self.terrain_points_x) * len(self.terrain_points_y)
        terrain = (InputField("terrain_heights", heights, "m,base-relative vertical"),) if heights else ()
        return (InputField("proprioception", self.observation_dim, "native observation units"),
                InputField("linear_velocity_B", 3, "m/s")) + terrain + (
                InputField("joint_torque", 12, "Nm,executed"),
                InputField("joint_acceleration", 12, "rad/s^2"),
                InputField("foot_contact_force_B", 12, "N"))

    @property
    def observation_dim(self): return sum(f.width for f in PROPRIOCEPTION_FIELDS)
    @property
    def state_dim(self): return sum(f.width for f in self.state_fields)
    @property
    def parameter_dim(self): return sum(f.width for f in self.parameter_fields)
    @property
    def privileged_dim(self): return self.state_dim + self.parameter_dim
    @property
    def load_dim(self): return sum(f.width for f in LOAD_FIELDS)

    def fields(self, group):
        return {"observations": PROPRIOCEPTION_FIELDS, "state": self.state_fields,
                "parameters": self.parameter_fields, "load_labels": LOAD_FIELDS}[group]

    def pack(self, group, values):
        """Pack explicitly named fields; reject missing/extra data rather than pad."""
        fields = self.fields(group)
        if set(values) != {f.name for f in fields}:
            raise ValueError(f"{group}: fields do not match declared schema")
        for f in fields:
            if values[f.name].ndim != 2 or values[f.name].shape[-1] != f.width:
                raise ValueError(f"{group}/{f.name}: expected [N,{f.width}]")
        return torch.cat([values[f.name] for f in fields], -1)

    def validate(self, group, value, num_envs, device=None):
        width = sum(f.width for f in self.fields(group))
        if value.shape != (num_envs, width) or value.dtype != torch.float32:
            raise ValueError(f"{group} must be float32 [{num_envs},{width}]")
        if device is not None and value.device != torch.device(device):
            raise ValueError(f"{group} must be on {device}")

    def resolved(self):
        return asdict(self)

    def contract(self):
        return dict(**self.resolved(), state_fields=[asdict(f) for f in self.state_fields],
                    normalization='identity/SI; absolute q, no nominal-angle subtraction',
                    actor_noise='shared physical noise on omega/gravity/q/dq only; cached per timestamp',
                    privileged_convention='clean, unclipped physical truth (including proprioception)',
                    previous_action='previous clipped issued position offset in policy units, before delay',
                    observation_fields=[asdict(f) for f in PROPRIOCEPTION_FIELDS],
                    load_fields=[asdict(f) for f in LOAD_FIELDS],
                    observation_dim=self.observation_dim, state_dim=self.state_dim,
                    parameter_dim=self.parameter_dim, load_dim=self.load_dim)
