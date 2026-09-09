"""Explicit migration of ABL3 checkpoints predating its separate GRF head."""
import warnings
import torch


def load_grf_decoders(algorithm, checkpoint):
    state = dict(checkpoint["decoder_state_dict"])
    if checkpoint.get("grf_decoder_state_dict") is None:
        warnings.warn("Legacy ABL3 checkpoint: removing GRF output rows; initializing separate GRF head and decoder optimizers fresh.")
        start = algorithm.privileged_grf_start_index
        for key in ("dec_out.weight", "dec_out.bias"):
            value = state[key]
            state[key] = torch.cat((value[:start], value[start+12:]), 0)
    else:
        algorithm.grf_decoder.load_state_dict(checkpoint["grf_decoder_state_dict"])
    algorithm.decoder.load_state_dict(state)
