"""
Convert PACT position-only checkpoint to full PACT format.

Usage:
  python convert_pact_pos_to_pact.py <checkpoint_path> <output_path>

Examples:
  python convert_pact_pos_to_pact.py ../../logs/go2_pact_pos_rough/Mar26_11-22-05_pact_pos_100hz/model_2000.pt pretained_checkpoints/rl_pos/go2_pact_pos_rough/model_2000_converted.pt
  python convert_pact_pos_to_pact.py ../../logs/a1_pact_pos_rough/Mar25_23-29-46_pact_pos_100hz/model_2000.pt pretained_checkpoints/rl_pos/a1_pact_pos_rough/model_2000_converted.pt
"""
import argparse
import torch
import os
from actor_critic_pact import ActorCritic_PACT, ContextDecoder
from actor_critic_pact_pos import ActorCritic_PACT_Pos

parser = argparse.ArgumentParser(description="Convert PACT pos-only checkpoint to full PACT")
parser.add_argument("checkpoint", help="Path to pos-only .pt checkpoint")
parser.add_argument("output", help="Path to save converted .pt checkpoint")
args = parser.parse_args()

# Load checkpoint
loaded_model = torch.load(args.checkpoint, map_location='cpu')
act_state_dict = loaded_model["model_state_dict"]
dec_state_dict = loaded_model["decoder_state_dict"]

for key in list(act_state_dict.keys()):
    act_state_dict[key.replace("_orig_mod.", "")] = act_state_dict.pop(key)
for key in list(dec_state_dict.keys()):
    dec_state_dict[key.replace("_orig_mod.", "")] = dec_state_dict.pop(key)

# Auto-detect layer sizes from checkpoint weights
enc_layers = [
    act_state_dict["context_encoder.ce_in.weight"].shape[0],     # e.g. 256
    act_state_dict["context_encoder.ce_h1.weight"].shape[0],     # e.g. 128
]
dec_layers = [
    dec_state_dict["dec_in.weight"].shape[0],                    # e.g. 128
    dec_state_dict["dec_h1.weight"].shape[0],                    # e.g. 256
]
print(f"Detected encoder layers: {enc_layers}, decoder layers: {dec_layers}")

# Load into pos-only model
actor_critic_pos = ActorCritic_PACT_Pos(57, 1030, 12, [512,256,128], [1024,256,128,64], 57*10, 16, 11, enc_layers, 'tanh', 1.0)
decoder_pos = ContextDecoder(27, dec_layers, 69)

actor_critic_pos.load_state_dict(act_state_dict)
decoder_pos.load_state_dict(dec_state_dict)

# Create full PACT model
actor_critic = ActorCritic_PACT(57, 1030, 12, [512,256,128], [1024,256,128,64], 57*10, 16, 11, enc_layers, 'tanh', 1.0)
decoder = ContextDecoder(27, dec_layers, 69)

# Copy sub-modules
actor_critic.act_trunk.load_state_dict(actor_critic_pos.act_trunk.state_dict())
actor_critic.act_pos_out.load_state_dict(actor_critic_pos.act_pos_out.state_dict())
actor_critic.act_tau_out.load_state_dict(actor_critic_pos.act_tau_out.state_dict())
actor_critic.critic.load_state_dict(actor_critic_pos.critic.state_dict())
decoder.load_state_dict(dec_state_dict)

# Copy encoder with key remapping (Sequential .0. -> bare Linear)
enc_sd = actor_critic_pos.context_encoder.state_dict()
remapped = {}
for key, val in enc_sd.items():
    new_key = key.replace("ce_out_var.0.", "ce_out_var.").replace("ce_velo_var.0.", "ce_velo_var.")
    remapped[new_key] = val
actor_critic.context_encoder.load_state_dict(remapped)

# Save
os.makedirs(os.path.dirname(args.output), exist_ok=True)
torch.save({
    "model_state_dict": actor_critic.state_dict(),
    "decoder_state_dict": decoder.state_dict(),
}, args.output)
print(f"Saved: {args.output}")
