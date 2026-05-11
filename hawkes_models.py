"""
Hawkes Process models with guaranteed non‑zero intensity.
"""
import numpy as np
from scipy.optimize import minimize

class HawkesExponential:
    """Univariate exponential kernel Hawkes process."""
    def __init__(self, decay=1.0):
        self.decay = decay
        self.mu = 0.01      # baseline intensity (small but >0)
        self.alpha = 0.1
        self.history = []

    def fit(self, events):
        event_times = np.where(events)[0].astype(float)
        if len(event_times) == 0:
            self.mu = 0.01
            self.alpha = 0.0
            return self
        T = len(events)

        def neg_log_lik(params):
            mu, alpha = params
            if mu <= 0 or alpha < 0 or alpha >= 1:
                return 1e10
            ll = 0.0
            for i, ti in enumerate(event_times):
                intensity = mu
                for tj in event_times[:i]:
                    intensity += alpha * self.decay * np.exp(-self.decay * (ti - tj))
                if intensity <= 0:
                    return 1e10
                ll += np.log(intensity)
            integral = mu * T
            for tj in event_times:
                integral += alpha * (1 - np.exp(-self.decay * (T - tj)))
            return -ll + integral

        res = minimize(neg_log_lik, [0.01, 0.5], bounds=[(1e-6, None), (0, 0.999)])
        self.mu, self.alpha = res.x
        self.history = event_times
        return self

    def predict_next_day_intensity(self, last_day_idx, total_days):
        t = last_day_idx + 1
        intensity = self.mu
        for tj in self.history:
            if tj < t:
                intensity += self.alpha * self.decay * np.exp(-self.decay * (t - tj))
        return max(intensity, 1e-6)

class HawkesSigned:
    """Simplified signed: use positive events only for upside intensity."""
    def __init__(self, decay=1.0):
        self.decay = decay
        self.model = HawkesExponential(decay=decay)

    def fit(self, pos_events, neg_events):
        # For upside intensity, we only care about positive jumps
        self.model.fit(pos_events)
        return self

    def predict_next_day_intensity(self, last_day_idx, total_days):
        return self.model.predict_next_day_intensity(last_day_idx, total_days)

class HawkesVolatilityExcited:
    def __init__(self, decay=1.0, window=20):
        self.decay = decay
        self.window = window
        self.model = HawkesExponential(decay=decay)

    def fit(self, returns):
        # Realised volatility as rolling standard deviation
        vol = returns.rolling(window=self.window).std()
        # Events: vol > 90th percentile of its own history
        thresh = vol.rolling(window=252, min_periods=50).quantile(0.9)
        thresh.fillna(vol.quantile(0.9), inplace=True)
        events = vol > thresh
        self.model.fit(events.values)
        return self

    def predict_next_day_intensity(self, last_day_idx):
        return self.model.predict_next_day_intensity(last_day_idx, None)

class HawkesEnsemble:
    def __init__(self, decay=1.0):
        self.exp = HawkesExponential(decay=decay)
        self.signed = HawkesSigned(decay=decay)
        self.vol = HawkesVolatilityExcited(decay=decay)

    def fit(self, returns, pos_events, neg_events):
        self.exp.fit(pos_events)
        self.signed.fit(pos_events, neg_events)
        self.vol.fit(returns)
        return self

    def predict_next_day_intensity(self, last_day_idx, total_days):
        i1 = self.exp.predict_next_day_intensity(last_day_idx, total_days)
        i2 = self.signed.predict_next_day_intensity(last_day_idx, total_days)
        i3 = self.vol.predict_next_day_intensity(last_day_idx)
        return (i1 + i2 + i3) / 3.0
