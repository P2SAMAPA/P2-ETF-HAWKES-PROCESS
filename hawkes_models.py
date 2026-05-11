"""
Hawkes Process models for jump contagion.
Implements:
- Exponential Jump Hawkes
- Signed Jump Hawkes (positive and negative processes)
- Volatility‑Excited Hawkes (events = volatility jumps)
- Ensemble Hawkes (average of above three)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import warnings

class HawkesExponential:
    """Univariate exponential kernel Hawkes process."""
    def __init__(self, decay=1.0):
        self.decay = decay          # beta (exponential rate)
        self.mu = None              # baseline intensity
        self.alpha = None           # self-excitation coefficient
        self.history = None

    def fit(self, events):
        """
        events: list of event times (indices) or binary array (1 = event)
        For daily data, events is a boolean array with True on jump days.
        """
        # Convert to event times (indices where event occurs)
        event_times = np.where(events)[0].astype(float)
        T = len(events)

        # Negative log-likelihood for exponential kernel
        def neg_log_lik(params):
            mu, alpha = params
            if mu <= 0 or alpha < 0 or alpha >= 1:
                return 1e10
            ll = 0.0
            for i, ti in enumerate(event_times):
                # Intensity at ti
                intensity = mu
                for tj in event_times[:i]:
                    intensity += alpha * self.decay * np.exp(-self.decay * (ti - tj))
                ll += np.log(intensity)
            # Integral term
            integral = mu * T
            for tj in event_times:
                integral += alpha * (1 - np.exp(-self.decay * (T - tj)))
            return -ll + integral

        res = minimize(neg_log_lik, [0.1, 0.5], bounds=[(1e-6, None), (0, 0.999)])
        self.mu, self.alpha = res.x
        self.history = event_times
        return self

    def intensity(self, time, event_times):
        """Compute intensity at given time (or array of times)."""
        if self.mu is None:
            return 0.0
        intensity = self.mu
        for tj in event_times:
            if tj < time:
                intensity += self.alpha * self.decay * np.exp(-self.decay * (time - tj))
        return intensity

    def predict_next_day_intensity(self, last_day_idx, total_days):
        """Predict intensity on next day (index last_day_idx+1)."""
        return self.intensity(last_day_idx + 1, self.history)


class HawkesSigned:
    """Bivariate Hawkes for positive and negative jumps."""
    def __init__(self, decay=1.0, cross_excite=True):
        self.decay = decay
        self.cross_excite = cross_excite
        self.mu_pos = None
        self.mu_neg = None
        self.alpha_pp = None   # pos -> pos
        self.alpha_pn = None   # pos -> neg (if cross)
        self.alpha_np = None   # neg -> pos
        self.alpha_nn = None   # neg -> neg

    def fit(self, pos_events, neg_events):
        """
        pos_events, neg_events: boolean arrays of same length.
        """
        pos_times = np.where(pos_events)[0].astype(float)
        neg_times = np.where(neg_events)[0].astype(float)
        T = len(pos_events)

        def intensity_pos(t):
            val = self.mu_pos
            for tj in pos_times:
                if tj < t:
                    val += self.alpha_pp * self.decay * np.exp(-self.decay * (t - tj))
            if self.cross_excite:
                for tj in neg_times:
                    if tj < t:
                        val += self.alpha_np * self.decay * np.exp(-self.decay * (t - tj))
            return val

        def intensity_neg(t):
            val = self.mu_neg
            for tj in neg_times:
                if tj < t:
                    val += self.alpha_nn * self.decay * np.exp(-self.decay * (t - tj))
            if self.cross_excite:
                for tj in pos_times:
                    if tj < t:
                        val += self.alpha_pn * self.decay * np.exp(-self.decay * (t - tj))
            return val

        # Negative log-likelihood (simplified: assume parameters independent for each process with cross-terms)
        # For brevity, we use a simpler approach: estimate univariate for positive and negative separately,
        # then add cross terms via MLE. Here we'll approximate by fitting separate Hawkes and then
        # estimate cross terms from co-occurrence. For production, use a full bivariate MLE.
        # For this engine, we'll use a pragmatic method:
        # 1. Fit univariate Hawkes on positive events -> mu_pos, alpha_pp
        # 2. Fit univariate on negative events -> mu_neg, alpha_nn
        # 3. Cross terms: alpha_pn = alpha_np = co-occurrence rate * average alpha
        pos_model = HawkesExponential(decay=self.decay).fit(pos_events)
        neg_model = HawkesExponential(decay=self.decay).fit(neg_events)
        self.mu_pos = pos_model.mu
        self.alpha_pp = pos_model.alpha
        self.mu_neg = neg_model.mu
        self.alpha_nn = neg_model.alpha
        # Cross: fraction of days where both pos and neg occur (should be zero by definition, but we approximate)
        both = np.logical_and(pos_events, neg_events).sum()
        total_events = len(pos_times) + len(neg_times)
        cross_rate = both / total_events if total_events > 0 else 0
        self.alpha_pn = cross_rate * self.alpha_pp
        self.alpha_np = cross_rate * self.alpha_nn
        self.pos_times = pos_times
        self.neg_times = neg_times
        return self

    def predict_next_day_intensity(self, last_day_idx, total_days):
        # For trading, we care about positive intensity (upside)
        t = last_day_idx + 1
        intensity = self.mu_pos
        for tj in self.pos_times:
            if tj < t:
                intensity += self.alpha_pp * self.decay * np.exp(-self.decay * (t - tj))
        if self.cross_excite:
            for tj in self.neg_times:
                if tj < t:
                    intensity += self.alpha_np * self.decay * np.exp(-self.decay * (t - tj))
        return intensity


class HawkesVolatilityExcited:
    """Events = large volatility jumps (Parkinson or realized vol)."""
    def __init__(self, decay=1.0):
        self.decay = decay
        self.model = HawkesExponential(decay=decay)

    def fit(self, returns, window=20):
        """
        Compute daily volatility (e.g., Parkinson) and detect jumps.
        """
        # Use realized volatility (absolute returns as proxy)
        vol = returns.abs().rolling(window=window).mean()
        # Define event when vol exceeds 2× its own rolling MAD
        mad = vol.rolling(window=252).apply(lambda x: np.median(np.abs(x - np.median(x))) * 1.4826, raw=False)
        threshold = 2 * mad
        events = vol > threshold
        self.model.fit(events.values)
        return self

    def predict_next_day_intensity(self, last_day_idx):
        return self.model.predict_next_day_intensity(last_day_idx, None)


class HawkesEnsemble:
    """Average of the three models (exponential, signed, volatility)."""
    def __init__(self, decay=1.0):
        self.exp = HawkesExponential(decay=decay)
        self.signed = HawkesSigned(decay=decay)
        self.vol = HawkesVolatilityExcited(decay=decay)

    def fit(self, returns, pos_events, neg_events, vol_events=None):
        """Fit all three sub-models."""
        self.exp.fit(pos_events)   # using positive jumps as events
        self.signed.fit(pos_events, neg_events)
        self.vol.fit(returns)
        return self

    def predict_next_day_intensity(self, last_day_idx, total_days):
        i1 = self.exp.predict_next_day_intensity(last_day_idx, total_days)
        i2 = self.signed.predict_next_day_intensity(last_day_idx, total_days)
        i3 = self.vol.predict_next_day_intensity(last_day_idx)
        return (i1 + i2 + i3) / 3.0
