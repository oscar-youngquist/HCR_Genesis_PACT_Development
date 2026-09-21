"""HardPACT-style conditional physics heads with explicit gradient boundaries."""

import math
import torch
from torch import nn
from .module_utils import init_weights


def mlp(input_dim, hidden, output_dim, activation):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), activation()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    result = nn.Sequential(*layers)
    result.apply(init_weights)
    return result


class B1Z1PhysicsDecoders(nn.Module):
    def __init__(self, latent_dim, force_layers, grf_layers, torque_scale=100., activation=nn.ELU):
        super().__init__()
        if not math.isfinite(torque_scale) or torque_scale <= 0:
            raise ValueError("GRF torque conditioning scale must be finite and positive")
        self.register_buffer("torque_scale", torch.tensor(float(torque_scale)))
        self.force = mlp(latent_dim + 14, force_layers, 9, activation)
        self.grf = mlp(latent_dim + 14 + 19, grf_layers, 12, activation)

    def predict_force(self, z, explicit):
        return self.force(torch.cat((z, explicit.detach()), -1))

    def predict_grf(self, z, explicit, nominal_torque):
        # Train the encoder through z, never the explicit estimator or torque
        # policy through these conditioning inputs. Torque is in physical Nm.
        if nominal_torque.shape[-1] != 19:
            raise ValueError("B1Z1 GRF conditioning requires all 19 joint torques")
        return self.grf(torch.cat((z, explicit.detach(), nominal_torque.detach() / self.torque_scale), -1))


def decode_context(model, context):
    raw = model.explicit_decoder(context["z"])
    explicit = torch.cat((raw[:, :6], raw[:, 6:10].sigmoid(), raw[:, 10:14]), -1)
    forces = model.physics_decoder.predict_force(context["z"], explicit)
    # Preserve the environment/label contract; forces now have their own head.
    return {**context, "explicit_condition": explicit,
            "explicit_prediction": torch.cat((raw[:, :6], forces, raw[:, 6:]), -1),
            "base_velocity": raw[:, :3], "ee_position": raw[:, 3:6],
            "base_wrench": forces[:, :6], "ee_force": forces[:, 6:9],
            "foot_contact_logits": raw[:, 6:10], "foot_height": raw[:, 10:14]}
