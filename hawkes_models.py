"""
Fast Hawkes‑like intensity using exponential weighting.
No fitting – just a heuristic that captures self‑excitation.
"""

import numpy as np
import pandas as pd

class FastHawkes:
    """
    Computes intensity as: baseline + decayed sum of past jumps.
    Uses exponential weighting (decay factor per day).
    """
    def __init__(self, baseline=0.01, alpha=0.5, decay=0.9):
        self.baseline = baseline      # mu
        self.alpha = alpha            # excitation strength
        self.decay = decay            # per‑day decay factor (0.9 = fast decay)

    def fit(self, events):
        """
        events: boolean array (True = jump)
        Pre‑computes intensity for each day (for training window).
        Returns intensity at each time step.
        """
        intensity = np.zeros(len(events))
        running = 0.0
        for i, is_jump in enumerate(events):
            running = self.decay * running
            if is_jump:
                running += self.alpha
            intensity[i] = self.baseline + running
        self.history_intensity = intensity
        return self

    def predict_next_day_intensity(self, last_day_idx, total_days=None):
        """Return the last computed intensity (for next day)."""
        if self.history_intensity is None or len(self.history_intensity) == 0:
            return self.baseline
        return self.history_intensity[-1]

# For compatibility with existing trainer code, we keep the same class names
HawkesExponential = FastHawkes

class HawkesSigned(FastHawkes):
    """Use same fast logic but only on positive jumps."""
    def fit(self, pos_events, neg_events=None):
        # For upside intensity, only positive jumps matter
        return super().fit(pos_events)

class HawkesVolatilityExcited(FastHawkes):
    """Same fast logic, applied to volatility jumps."""
    def fit(self, returns):
        # Compute volatility jumps (e.g., 90th percentile of rolling vol)
        vol = returns.rolling(20).std()
        thresh = vol.rolling(252, min_periods=50).quantile(0.9).fillna(vol.quantile(0.9))
        events = (vol > thresh).values
        return super().fit(events)

class HawkesEnsemble:
    def __init__(self, baseline=0.01, alpha=0.5, decay=0.9):
        self.exp = FastHawkes(baseline, alpha, decay)
        self.signed = FastHawkes(baseline, alpha, decay)
        self.vol = FastHawkes(baseline, alpha, decay)

    def fit(self, returns, pos_events, neg_events):
        self.exp.fit(pos_events)
        self.signed.fit(pos_events)   # signed only cares about positive
        self.vol.fit(returns)
        return self

    def predict_next_day_intensity(self, last_day_idx, total_days):
        i1 = self.exp.predict_next_day_intensity(last_day_idx, total_days)
        i2 = self.signed.predict_next_day_intensity(last_day_idx, total_days)
        i3 = self.vol.predict_next_day_intensity(last_day_idx)
        return (i1 + i2 + i3) / 3.0
