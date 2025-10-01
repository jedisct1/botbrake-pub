"""
Statistical models for abuse detection (§4).
Implements Beta-Binomial error model, robust z-scores, and rate anomaly detection (EWMA/CUSUM).
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BetaBinomialPrior:
    """
    Beta-Binomial prior parameters (§4.1, Appendix A).
    Represents learned baseline from global traffic.
    """

    alpha: float = 1.0  # Weak prior
    beta: float = 1.0  # Weak prior

    @classmethod
    def from_error_rates(cls, error_rates: List[float]) -> "BetaBinomialPrior":
        """
        Fit prior from observed error rates using method of moments (Appendix A).

        Args:
            error_rates: List of error rates (0.0 to 1.0) from legitimate entities

        Returns:
            BetaBinomialPrior with fitted alpha and beta
        """
        if not error_rates:
            return cls()  # Default weak prior

        arr = np.array(error_rates)
        mean = np.mean(arr)
        var = np.var(arr)

        # Avoid division by zero
        if var <= 0 or mean <= 0 or mean >= 1:
            return cls()

        # Method of moments: μ = α/(α+β), v = μ(1-μ)/(α+β+1)
        # Solve for α, β
        # α+β = μ(1-μ)/v - 1
        sum_params = mean * (1 - mean) / var - 1
        if sum_params <= 0:
            return cls()

        alpha = mean * sum_params
        beta = (1 - mean) * sum_params

        # Regularization: ensure reasonable bounds
        alpha = max(0.5, min(alpha, 1000))
        beta = max(0.5, min(beta, 1000))

        return cls(alpha=alpha, beta=beta)


class BetaBinomialModel:
    """
    Beta-Binomial error model for entity error rate analysis (§4.1).
    """

    def __init__(self, prior: BetaBinomialPrior):
        """
        Initialize model with prior.

        Args:
            prior: Beta-Binomial prior parameters
        """
        self.prior = prior

    def posterior_probability_exceeds(self, errors: int, total: int, threshold: float) -> float:
        """
        Calculate P(θ_E > threshold) given observations (§4.1).

        Args:
            errors: Number of error responses (4xx + 5xx)
            total: Total number of requests
            threshold: Error rate threshold to test against

        Returns:
            Probability that true error rate exceeds threshold
        """
        if total == 0:
            return 0.0

        # Posterior is Beta(α0 + errors, β0 + (total - errors))
        alpha_post = self.prior.alpha + errors
        beta_post = self.prior.beta + (total - errors)

        # P(θ > threshold) = 1 - CDF(threshold)
        return 1.0 - stats.beta.cdf(threshold, alpha_post, beta_post)

    def posterior_mean(self, errors: int, total: int) -> float:
        """
        Calculate posterior mean error rate θ̂_E.

        Args:
            errors: Number of error responses
            total: Total number of requests

        Returns:
            Posterior mean error rate
        """
        alpha_post = self.prior.alpha + errors
        beta_post = self.prior.beta + (total - errors)
        return alpha_post / (alpha_post + beta_post)

    def credible_interval(
        self, errors: int, total: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate credible interval for error rate.

        Args:
            errors: Number of error responses
            total: Total number of requests
            confidence: Confidence level (default 95%)

        Returns:
            (lower_bound, upper_bound) credible interval
        """
        alpha_post = self.prior.alpha + errors
        beta_post = self.prior.beta + (total - errors)

        lower = (1 - confidence) / 2
        upper = 1 - lower

        return (
            stats.beta.ppf(lower, alpha_post, beta_post),
            stats.beta.ppf(upper, alpha_post, beta_post),
        )


def compute_mad(values: List[float]) -> float:
    """
    Compute Median Absolute Deviation (MAD) (§4.2).

    Args:
        values: List of numeric values

    Returns:
        MAD value
    """
    if not values:
        return 0.0

    arr = np.array(values)
    median = np.median(arr)
    deviations = np.abs(arr - median)
    return np.median(deviations)


def robust_z_score(value: float, values: List[float]) -> float:
    """
    Compute robust z-score using MAD (§4.2).
    rz = (value - median) / (1.4826 * MAD)

    The constant 1.4826 makes MAD a consistent estimator of standard deviation
    for normally distributed data.

    Args:
        value: Value to test
        values: Reference distribution

    Returns:
        Robust z-score
    """
    if not values:
        return 0.0

    arr = np.array(values)
    median = np.median(arr)
    mad = compute_mad(values)

    # Avoid division by zero
    if mad < 1e-9:
        return 0.0

    return (value - median) / (1.4826 * mad)


def is_anomalous_by_mad(value: float, values: List[float], threshold: float = 4.0) -> bool:
    """
    Test if a value is anomalous using MAD-based robust z-score (§4.2).

    Args:
        value: Value to test
        values: Reference distribution
        threshold: Z-score threshold (default 4.0 from §10)

    Returns:
        True if anomalous
    """
    rz = robust_z_score(value, values)
    return abs(rz) > threshold


class EWMATracker:
    """
    Exponentially Weighted Moving Average tracker for rate anomaly detection (§4.3).
    """

    def __init__(self, alpha: float = 0.3):
        """
        Initialize EWMA tracker.

        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive to recent changes.
                  Default 0.3 per §4.3.
        """
        self.alpha = alpha
        self.ewma: Optional[float] = None
        self.history: List[float] = []  # For computing std dev

    def update(self, value: float):
        """
        Update EWMA with new observation.

        Args:
            value: New rate observation (e.g., requests per second)
        """
        if self.ewma is None:
            self.ewma = value
        else:
            self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma

        self.history.append(value)
        # Keep bounded history (last 100 observations)
        if len(self.history) > 100:
            self.history.pop(0)

    def get_deviation(self, value: float) -> float:
        """
        Get deviation of value from EWMA.

        Args:
            value: Value to test

        Returns:
            Absolute deviation from EWMA
        """
        if self.ewma is None:
            return 0.0
        return abs(value - self.ewma)

    def get_std(self) -> float:
        """
        Get standard deviation of historical rates.

        Returns:
            Standard deviation (0 if insufficient history)
        """
        if len(self.history) < 2:
            return 0.0
        return float(np.std(self.history))

    def compute_tail_probability(self, value: float) -> float:
        """
        Compute tail probability P(rate >= value) using Poisson/NegBin model (§4.3).

        Args:
            value: Rate to test

        Returns:
            Tail probability (0-1). Lower = more anomalous.
        """
        if self.ewma is None or self.ewma <= 0:
            return 1.0

        # Use Poisson model with EWMA as lambda
        # For overdispersion, could use NegBin, but Poisson is simpler for MVP
        try:
            p_tail = 1.0 - stats.poisson.cdf(value, self.ewma)
            return max(0.0, min(1.0, p_tail))
        except (ValueError, RuntimeWarning):
            return 1.0


class CUSUMTracker:
    """
    Cumulative Sum (CUSUM) tracker for detecting persistent rate shifts (§4.3).
    """

    def __init__(self, k: float = 2.0, h: float = 5.0, baseline_std: float = 1.0):
        """
        Initialize CUSUM tracker.

        Args:
            k: Slack parameter (threshold for shift detection), in units of std dev.
               Default 2.0 per §4.3.
            h: Decision threshold (cumulative sum trigger), in units of std dev.
               Default 5.0 per §4.3.
            baseline_std: Baseline standard deviation of rates.
        """
        self.k = k
        self.h = h
        self.baseline_std = baseline_std
        self.s_t: float = 0.0  # Cumulative sum
        self.baseline_rate: Optional[float] = None

    def set_baseline(self, rate: float, std: float):
        """
        Set baseline rate and std deviation.

        Args:
            rate: Baseline rate (e.g., from EWMA)
            std: Standard deviation
        """
        self.baseline_rate = rate
        self.baseline_std = max(std, 1e-6)  # Avoid division by zero

    def update(self, value: float) -> bool:
        """
        Update CUSUM with new observation.

        Args:
            value: New rate observation

        Returns:
            True if CUSUM threshold exceeded (shift detected)
        """
        if self.baseline_rate is None:
            self.baseline_rate = value
            return False

        # S_t = max(0, S_{t-1} + (rate - baseline - k*std))
        deviation = (value - self.baseline_rate) / self.baseline_std
        self.s_t = max(0.0, self.s_t + deviation - self.k)

        # Trigger if S_t > h
        return self.s_t > self.h

    def reset(self):
        """Reset CUSUM state (after detecting shift)."""
        self.s_t = 0.0

    def get_score(self) -> float:
        """
        Get current CUSUM score (0-100).

        Returns:
            Score based on S_t relative to threshold h
        """
        if self.h <= 0:
            return 0.0
        return min(100.0, 100.0 * self.s_t / self.h)


class RateAnomalyDetector:
    """
    Combined rate anomaly detector using EWMA and CUSUM (§4.3).
    """

    def __init__(self, alpha: float = 0.3, cusum_k: float = 2.0, cusum_h: float = 5.0):
        """
        Initialize rate anomaly detector.

        Args:
            alpha: EWMA smoothing factor
            cusum_k: CUSUM slack parameter
            cusum_h: CUSUM decision threshold
        """
        self.ewma = EWMATracker(alpha)
        self.cusum = CUSUMTracker(cusum_k, cusum_h)

    def update(self, rate: float):
        """
        Update detector with new rate observation.

        Args:
            rate: Request rate (e.g., requests per second)
        """
        self.ewma.update(rate)

        # Update CUSUM with latest EWMA baseline
        if self.ewma.ewma is not None:
            self.cusum.set_baseline(self.ewma.ewma, self.ewma.get_std())

        # Check if shift detected
        if self.cusum.update(rate):
            # Shift detected - could reset or escalate
            pass  # For now, just accumulate

    def compute_burst_signal(self, current_rate: float, gamma: float = 0.5) -> float:
        """
        Compute burst signal S_burst (§6.1).

        S_burst = 100 * (1 - p_tail)^γ

        Args:
            current_rate: Current request rate
            gamma: Dampening exponent (default 0.5 per §4.3)

        Returns:
            Burst signal score (0-100)
        """
        p_tail = self.ewma.compute_tail_probability(current_rate)
        return 100.0 * ((1.0 - p_tail) ** gamma)

    def is_anomalous(self, current_rate: float, p_threshold: float = 1e-6) -> bool:
        """
        Check if current rate is anomalous.

        Args:
            current_rate: Current request rate
            p_threshold: Tail probability threshold (default 1e-6 per §4.3)

        Returns:
            True if anomalous
        """
        p_tail = self.ewma.compute_tail_probability(current_rate)
        return p_tail < p_threshold


class BaselineStatistics:
    """
    Stores baseline statistics learned from training phase.
    """

    def __init__(self):
        """Initialize baseline statistics."""
        self.global_error_rate: Optional[float] = None
        self.beta_prior: BetaBinomialPrior = BetaBinomialPrior()

        # Distribution baselines for exploration metrics
        self.explore_ratio_values: List[float] = []
        self.prefix_fanout_values: Dict[int, List[float]] = {}  # depth -> values

        # Depth score baselines (§2.3)
        self.depth_score_values: List[float] = []

    def set_global_error_rate(self, rate: float):
        """Set global (legitimate) error rate baseline."""
        self.global_error_rate = rate

    def set_beta_prior(self, prior: BetaBinomialPrior):
        """Set Beta-Binomial prior."""
        self.beta_prior = prior

    def add_explore_ratio(self, ratio: float):
        """Add exploration ratio observation for baseline."""
        self.explore_ratio_values.append(ratio)

    def add_prefix_fanout(self, depth: int, fanout: float):
        """Add prefix fanout observation for baseline."""
        if depth not in self.prefix_fanout_values:
            self.prefix_fanout_values[depth] = []
        self.prefix_fanout_values[depth].append(fanout)

    def get_explore_ratio_rz(self, value: float) -> float:
        """Get robust z-score for exploration ratio."""
        return robust_z_score(value, self.explore_ratio_values)

    def get_prefix_fanout_rz(self, depth: int, value: float) -> float:
        """Get robust z-score for prefix fanout at depth."""
        if depth not in self.prefix_fanout_values:
            return 0.0
        return robust_z_score(value, self.prefix_fanout_values[depth])

    def add_depth_score(self, score: float):
        """Add depth score observation for baseline (§2.3)."""
        self.depth_score_values.append(score)

    def get_depth_score_rz(self, value: float) -> float:
        """Get robust z-score for depth score (§2.3)."""
        return robust_z_score(value, self.depth_score_values)


if __name__ == "__main__":
    # Test Beta-Binomial model
    print("Testing Beta-Binomial Model:")

    # Fit prior from legitimate traffic (low error rates)
    legit_error_rates = [0.05, 0.08, 0.06, 0.07, 0.09, 0.04]
    prior = BetaBinomialPrior.from_error_rates(legit_error_rates)
    print(f"Fitted prior: α={prior.alpha:.2f}, β={prior.beta:.2f}")

    model = BetaBinomialModel(prior)

    # Test case: entity with high error rate
    errors = 80
    total = 100
    theta_legit = 0.08
    delta = 0.10

    prob = model.posterior_probability_exceeds(errors, total, theta_legit + delta)
    print(f"P(θ > {theta_legit + delta}) = {prob:.4f}")
    print(f"Posterior mean: {model.posterior_mean(errors, total):.3f}")

    # Test robust z-scores
    print("\nTesting Robust Z-scores:")
    values = [0.1, 0.12, 0.11, 0.13, 0.09, 0.11]
    outlier = 0.8
    rz = robust_z_score(outlier, values)
    print(f"Robust z-score for {outlier}: {rz:.2f}")
    print(f"Anomalous? {is_anomalous_by_mad(outlier, values)}")
