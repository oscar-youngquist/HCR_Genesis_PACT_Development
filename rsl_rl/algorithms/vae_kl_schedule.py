"""Stateless, resume-exact VAE KL weighting, independent of policy PPO KL."""
import math


def cosine_vae_beta(iteration, initial, final, start, duration):
    """Clamp outside [start,start+duration]; duration=0 means constant final."""
    if duration < 0:
        raise ValueError("vae_kl_warmup_iterations must be nonnegative")
    if duration == 0:
        return float(final)
    p = min(max((float(iteration)-start)/duration,0.0),1.0)
    return initial + .5*(final-initial)*(1.0-math.cos(math.pi*p))
