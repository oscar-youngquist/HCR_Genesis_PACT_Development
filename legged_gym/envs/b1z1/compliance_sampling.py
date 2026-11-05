"""Reset-only profile sampling. IDs are diagnostics, never actor inputs.

Enumerate the 2^12 activation masks once on CPU, then sample compatible masks
and magnitudes in batches on the target device. No rejection loops or changes
to force distributions, reference filters, or handoff gates are needed.
"""
import warnings

import torch


FAMILY_NAMES = ("rejection", "arm_only", "base_first", "handoff", "asymmetric", "uniform", "fixed")


class ProfileSampler:
    def __init__(self, cfg, device, dtype=torch.float32):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        low = torch.tensor(cfg.ee_profile_low + cfg.base_profile_low, dtype=torch.float64)
        high = torch.tensor(cfg.ee_profile_high + cfg.base_profile_high, dtype=torch.float64)
        self.low, self.high = low.to(device=device, dtype=dtype), high.to(device=device, dtype=dtype)
        self.diagnostics = []
        self.mode = cfg.sampling_mode
        if self.mode not in ("uniform", "mixture"):
            raise ValueError("sampling_mode must be uniform or mixture")
        self.fixed = bool((low == high).all())
        self.effective_weights = torch.zeros(7, device=device, dtype=dtype)
        if self.mode == "uniform" or self.fixed:
            self.effective_weights[5 if self.mode == "uniform" else 6] = 1
            return

        def vector(value, size, name, probability=False):
            out = torch.as_tensor(value, dtype=torch.float64)
            if out.shape != (size,) or not torch.isfinite(out).all() or (out < 0).any():
                raise ValueError(f"Invalid compliance sampling {name}")
            if probability and (out > 1).any():
                raise ValueError(f"{name} must be in [0,1]")
            return out

        weights = vector(cfg.family_weights, 6, "family_weights")
        if weights.sum() == 0:
            raise ValueError("At least one compliance family weight must be positive")
        probs = vector(cfg.ee_axis_probability, 3, "ee_axis_probability", True)
        probs = torch.cat((probs, vector(cfg.base_axis_probability, 3, "base_axis_probability", True)))
        for key in ("paired_direction_probability", "symmetric_gain_probability",
                    "planar_gain_correlation_probability", "payload_probability", "reverse_payload_probability"):
            vector([getattr(cfg, key)], 1, key, True)
        ranges = {}
        for key in ("ee_active_range", "base_active_range", "payload_low_range",
                    "payload_high_range", "payload_planar_range"):
            r = vector(getattr(cfg, key), 2, key, True)
            if r[0] <= 0 or r[0] > r[1]:
                raise ValueError(f"{key} must be an ordered strictly positive range")
            ranges[key] = r
        if ranges['payload_low_range'][1] >= ranges['payload_high_range'][0]:
            raise ValueError("Payload low/high ranges must be strictly separated")

        masks = ((torch.arange(4096)[:, None] >> torch.arange(12)) & 1).bool()
        pairs = masks.reshape(-1, 6, 2)
        active = pairs.any(-1)
        both = pairs.all(-1)
        p_pair = cfg.paired_direction_probability
        mask_weights = torch.where(~active, 1-probs,
            probs * torch.where(both, p_pair, (1-p_pair)/2)).prod(-1)
        ee, base = masks[:, :6], masks[:, 6:]
        # Handoff shares signed horizontal directions, not gain magnitudes.
        # Vertical EE yielding has no direct base-z counterpart; yaw is independent.
        selectors = [~masks.any(-1), ee.any(-1) & ~base.any(-1),
            ~ee.any(-1) & base.any(-1),
            (ee[:, :4] == base[:, :4]).all(-1) & ee[:, :4].any(-1),
            masks.any(-1)]
        # Structural zeros (e.g. a stiff arm in base-first) are mandatory, not
        # Bernoulli axis draws; axis probabilities apply only to available groups.
        group_weights = [torch.ones_like(mask_weights),
            torch.where(~active[:, :3], 1-probs[:3], probs[:3]*torch.where(both[:, :3], p_pair, (1-p_pair)/2)).prod(-1),
            torch.where(~active[:, 3:], 1-probs[3:], probs[3:]*torch.where(both[:, 3:], p_pair, (1-p_pair)/2)).prod(-1),
            mask_weights, mask_weights]
        self.tables = []
        families, table_weights = [], []
        active_range = torch.cat((ranges['ee_active_range'].repeat(6, 1), ranges['base_active_range'].repeat(6, 1)))
        for family in range(5):
            variants = [("general", 1.)]
            if family == 4:
                p, r = cfg.payload_probability, cfg.reverse_payload_probability
                variants = [("general", 1-p), ("payload", p*(1-r)), ("reverse_payload", p*r)]
            valid_tables = []
            for variant, probability in variants:
                if probability == 0:
                    continue
                limits = active_range.clone()
                selector = selectors[family].clone()
                if variant != "general":
                    limits[:4] = ranges['payload_planar_range']
                    limits[4] = ranges['payload_low_range' if variant == 'payload' else 'payload_high_range']
                    limits[5] = ranges['payload_high_range' if variant == 'payload' else 'payload_low_range']
                    selector &= ee[:, 4:].all(-1) & ee[:, :4].any(-1)
                lo, hi = torch.maximum(low, limits[:, 0]), torch.minimum(high, limits[:, 1])
                # Explicit fixed coefficients override generic magnitude ranges,
                # but not the semantic low/high ranges of payload presets.
                if variant == 'general':
                    lo = torch.where(low == high, low, lo)
                    hi = torch.where(low == high, high, hi)
                lower = torch.where(masks, lo, 0.)
                upper = torch.where(masks, hi, 0.)
                compatible = ((lower >= low) & (upper <= high) & (upper >= lower)
                              & (~masks | (upper > 0))).all(-1)
                if family == 4:
                    l, h = lower.reshape(-1, 6, 2), upper.reshape(-1, 6, 2)
                    # At least one opposing pair must be capable of inequality.
                    compatible &= ((l[..., 0] != h[..., 1]) | (h[..., 0] != l[..., 1])).any(-1)
                valid = selector & compatible & (group_weights[family] > 0)
                if not valid.any():
                    if weights[family] > 0:
                        self.diagnostics.append(f"Excluded {FAMILY_NAMES[family]}/{variant}: incompatible bounds or axis probabilities")
                    continue
                idx = len(self.tables)
                self.tables.append((lower[valid], upper[valid], group_weights[family][valid]))
                valid_tables.append((idx, probability))
            if not valid_tables:
                weights[family] = 0
            else:
                total = sum(p for _, p in valid_tables)
                for idx, p in valid_tables:
                    families.append((family, idx))
                    table_weights.append(float(weights[family])*p/total)
        for message in self.diagnostics:
            warnings.warn(message + "; renormalizing remaining mixture weights", UserWarning, stacklevel=2)
        if weights.sum() == 0:
            raise ValueError("No compatible positive-weight compliance families; adjust bounds/weights or use uniform")
        self.effective_weights[:6] = (weights/weights.sum()).to(device=device, dtype=dtype)
        # Flatten the family->pattern distribution for one batched categorical
        # draw at reset. No device-to-host row counts or per-family GPU syncs.
        lows, highs, probabilities, family_ids = [], [], [], []
        for (family, table), weight in zip(families, table_weights):
            if weight == 0:
                continue
            lo, hi, probability = self.tables[table]
            lows.append(lo); highs.append(hi)
            probabilities.append(probability/probability.sum()*weight)
            family_ids.append(torch.full((len(lo),), family, dtype=torch.long))
        if weights[5] > 0:
            lows.append(low[None]); highs.append(high[None])
            probabilities.append(weights[5:6]); family_ids.append(torch.tensor([5]))
        self.pattern_low, self.pattern_high, self.pattern_weights = (
            torch.cat(items).to(device=device, dtype=dtype) for items in (lows, highs, probabilities))
        self.pattern_families = torch.cat(family_ids).to(device=device)
        del self.tables

    def sample(self, n):
        # Same draw shape/order as legacy uniform, including fixed coefficients.
        uniform = self.low + (self.high-self.low)*torch.rand(n, 12, device=self.device)
        if self.mode == 'uniform' or self.fixed or n == 0:
            family = 5 if self.mode == 'uniform' else 6
            return uniform, torch.full((n,), family, device=self.device, dtype=torch.long)
        index = torch.multinomial(self.pattern_weights, n, replacement=True)
        lo, hi, ids = self.pattern_low[index], self.pattern_high[index], self.pattern_families[index]
        u = torch.rand(n, 12, device=self.device, dtype=self.dtype)
        structured = (ids < 4)[:, None]
        # Shared quantiles correlate planar gains while respecting each bound;
        # EE/base draws stay independent even in handoff profiles.
        for start in (0, 6):
            correlated = structured & (torch.rand(n, 1, device=self.device) < self.cfg.planar_gain_correlation_probability)
            u[:, start:start+4] = torch.where(correlated, u[:, start:start+1], u[:, start:start+4])
        l, h, quantile = lo.reshape(-1, 6, 2), hi.reshape(-1, 6, 2), u.reshape(-1, 6, 2)
        common_low, common_high = l.amax(-1), h.amin(-1)
        symmetric = structured & (torch.rand(n, 6, device=self.device) < self.cfg.symmetric_gain_probability)
        symmetric &= (common_high >= common_low) & (l > 0).all(-1)
        shared = common_low + (common_high-common_low)*quantile[..., 0]
        values = torch.where(symmetric[..., None], shared[..., None], l+(h-l)*quantile).flatten(1)
        return torch.where((ids == 5)[:, None], uniform, values), ids

    def statistics(self, profiles, ids):
        """Statistics of newly drawn episodes, not length-biased resident robots."""
        stats = {f"compliance_sampling/{name}_fraction": (ids == i).float().mean()
                 for i, name in enumerate(FAMILY_NAMES)}
        stats.update({f"compliance_sampling/{name}_effective_weight": self.effective_weights[i]
                      for i, name in enumerate(FAMILY_NAMES)})
        stats['compliance_sampling/zero_coefficient_fraction'] = (profiles == 0).float().mean()
        stats['compliance_sampling/ee_mean'] = profiles[:, :6].mean()
        stats['compliance_sampling/base_mean'] = profiles[:, 6:].mean()
        stats['compliance_sampling/yaw_disabled_fraction'] = (profiles[:, 10:] == 0).all(-1).float().mean()
        stats['compliance_sampling/draw_count'] = profiles.new_tensor(len(ids))
        return stats
