"""BR observation noise in native SI units, independent of simulator/history.

Shared noise amplitudes are physical; HardPACT's observation scale multipliers
are intentionally absent because BR does not normalize these channels.
"""
import torch


class BRActorObservationCache:
    def __init__(self, num_envs, device, noise_cfg, clip, seed):
        self.generator=torch.Generator(device=device).manual_seed(int(seed)+104729)
        self.values=torch.zeros(num_envs,45,device=device)
        self.noise=torch.zeros_like(self.values)
        self.amplitude=torch.zeros(45,device=device)
        self.clip=float(clip)
        self.enabled=bool(noise_cfg.add_noise)
        if self.enabled:
            s=noise_cfg.noise_scales; level=noise_cfg.noise_level
            for sl,value in ((slice(0,3),s.ang_vel),(slice(3,6),s.gravity),
                             (slice(9,21),s.dof_pos),(slice(21,33),s.dof_vel)):
                self.amplitude[sl]=level*value
            if not torch.isfinite(self.amplitude).all() or (self.amplitude<0).any():
                raise ValueError('BR noise amplitudes must be finite and nonnegative')

    def refresh(self, clean, ids=None, *, redraw=True):
        """Called only when stepping/resetting; observe() merely reads values.

        Command refresh at the same physics timestamp reuses sensor noise.
        Selected resets redraw only those rows, preserving every other cache.
        """
        ids=slice(None) if ids is None else ids
        if redraw and self.enabled:
            self.noise[ids]=(2*torch.rand(self.values[ids].shape,device=self.values.device,
                                         generator=self.generator)-1)*self.amplitude
        self.values[ids]=(clean[ids]+self.noise[ids]).clamp(-self.clip,self.clip)

    def state_dict(self): return {'rng':self.generator.get_state().clone()}
    def load_state_dict(self,state): self.generator.set_state(state['rng'].cpu())
