"""
Scoring function for abuse detection (§6).
Combines multiple signals with weights and dampeners to produce 0-100 score.
"""

from dataclasses import dataclass
from botbrake.config import ScoringWeights


@dataclass
class SignalComponents:
    """Individual signal components (§6.1)."""

    # Signal scores (0-100 each)
    s_err: float = 0.0  # Error propensity
    s_explore: float = 0.0  # Exploration/scanning abnormality (HIGH exploration)
    s_hammer: float = 0.0  # Hammering/concentration (LOW exploration, high volume)
    s_burst: float = 0.0  # Rate anomaly (EWMA/CUSUM)
    s_persist: float = 0.0  # Persistence across windows
    s_spread: float = 0.0  # Spread across entities
    s_cross: float = 0.0  # Cross-entity corroboration
    s_dominance: float = 0.0  # Network dominance (disproportionate traffic share)

    # Dampener scores (0-100 each, subtracted)
    d_new: float = 0.0  # New content leniency
    d_volume: float = 0.0  # Low volume uncertainty
    d_legit_ua: float = 0.0  # Known good crawler


@dataclass
class ScoringResult:
    """Result of scoring calculation."""

    final_score: float  # 0-100
    components: SignalComponents
    explanation: str
    recommend_block: bool


class Scorer:
    """
    Abuse scoring engine (§6).
    """

    def __init__(self, weights: ScoringWeights):
        """
        Initialize scorer with weights.

        Args:
            weights: Scoring weights configuration
        """
        self.weights = weights

    def compute_error_signal(
        self, posterior_prob: float, posterior_mean: float, theta_legit: float, delta: float
    ) -> float:
        """
        Compute error propensity signal S_err (§6.1).

        S_err = 100 * clamp01((θ̂_E - θ_legit - δ0) / Δ_err) * posterior_confidence

        Args:
            posterior_prob: P(θ_E > θ_legit + δ)
            posterior_mean: θ̂_E (posterior mean error rate)
            theta_legit: Baseline legitimate error rate
            delta: Tolerance margin

        Returns:
            Error signal score (0-100)
        """
        if theta_legit < 0:
            return 0.0

        # Distance beyond acceptable threshold
        delta_err = 0.20  # Normalize to 20 percentage points as full scale
        excess = posterior_mean - theta_legit - delta
        normalized = max(0.0, min(1.0, excess / delta_err))

        # Weight by posterior confidence
        return 100.0 * normalized * posterior_prob

    def compute_exploration_signal(self, max_robust_z: float) -> float:
        """
        Compute exploration/scanning signal S_explore (§6.1).

        Uses logistic function: S_explore = 100 / (1 + exp(-a*(rz - b)))

        Args:
            max_robust_z: Maximum robust z-score among exploration metrics

        Returns:
            Exploration signal score (0-100)
        """
        # Logistic parameters: inflection at rz=4, steepness a=0.5
        a = 0.5
        b = 4.0

        import math

        return 100.0 / (1.0 + math.exp(-a * (max_robust_z - b)))

    def compute_hammer_signal(
        self, top_path_ratio: float, total_requests: int, n_hammer_min: int = 500
    ) -> float:
        """
        Compute hammering/concentration signal S_hammer (§6.1, §4.2b).

        S_hammer = 100 * clamp01((top_path_ratio - 0.5) / 0.4) if n >= n_hammer_min

        Detects low-diversity, high-volume abuse patterns.

        Args:
            top_path_ratio: Ratio of requests to most frequent path (0-1)
            total_requests: Total number of requests
            n_hammer_min: Minimum requests required (default 500 per §4.2b)

        Returns:
            Hammering signal score (0-100)
        """
        # Guard against low-volume false positives
        if total_requests < n_hammer_min:
            return 0.0

        # Linear ramp from 0.5 to 0.9
        # 0.5 = 0 score (legitimate single-page focus)
        # 0.9 = 100 score (extreme hammering)
        normalized = (top_path_ratio - 0.5) / 0.4
        clamped = max(0.0, min(1.0, normalized))
        return 100.0 * clamped

    def compute_persistence_signal(self, num_windows: int) -> float:
        """
        Compute persistence signal S_persist (§6.1).

        S_persist = 20 * num_consecutive_windows, clipped to 0-100

        Args:
            num_windows: Number of consecutive windows with signals

        Returns:
            Persistence signal score (0-100)
        """
        return min(100.0, 20.0 * num_windows)

    def compute_spread_signal(self, unique_count: int, velocity: float) -> float:
        """
        Compute spread signal S_spread (§6.1).

        For UA/CIDR/ref/path: abnormal spread across IPs or prefixes.

        Args:
            unique_count: Number of unique entities (IPs, UAs, etc.)
            velocity: Rate of spread

        Returns:
            Spread signal score (0-100)
        """
        # Heuristic: high unique count in short time is suspicious
        # E.g., >200 IPs for a UA in 10 minutes
        if unique_count > 200:
            return min(100.0, (unique_count - 200) / 5.0)
        return 0.0

    def compute_dominance_signal(
        self, entity_requests: int, total_requests: int, dominance_threshold: float = 0.30
    ) -> float:
        """
        Compute network dominance signal S_dominance.

        Detects when a single entity (network, UA, etc.) represents a
        disproportionate share of total traffic, indicating coordinated abuse.

        Args:
            entity_requests: Number of requests from this entity
            total_requests: Total requests across all entities
            dominance_threshold: Minimum ratio to be considered dominant (default 0.30 = 30%)

        Returns:
            Dominance signal score (0-100)
        """
        if total_requests == 0:
            return 0.0

        traffic_ratio = entity_requests / total_requests

        # No signal if below threshold
        if traffic_ratio < dominance_threshold:
            return 0.0

        # Linear ramp from threshold (30%) to extreme dominance (60%)
        # Tighter range than original 80% to make detection more aggressive
        # 30% = 0 score, 60% = 100 score
        normalized = (traffic_ratio - dominance_threshold) / (0.60 - dominance_threshold)
        clamped = max(0.0, min(1.0, normalized))
        return 100.0 * clamped

    def compute_volume_dampener(self, total_requests: int, n_min: int) -> float:
        """
        Compute low volume dampener D_volume (§6.2).

        Reduces score when evidence is insufficient.

        Args:
            total_requests: Number of requests observed
            n_min: Minimum required for confident decision

        Returns:
            Dampener value (0-40)
        """
        if total_requests >= n_min:
            return 0.0

        # Linear ramp: full dampening at 0 requests
        ratio = total_requests / n_min if n_min > 0 else 1.0
        return 40.0 * (1.0 - ratio)

    def compute_new_content_dampener(self, is_new_content: bool, grace_applies: bool) -> float:
        """
        Compute new content dampener D_new (§6.2).

        Args:
            is_new_content: Whether this entity touches new content
            grace_applies: Whether grace period applies

        Returns:
            Dampener value (0-30)
        """
        if is_new_content and grace_applies:
            return 30.0
        return 0.0

    def compute_final_score(self, components: SignalComponents) -> float:
        """
        Compute final score from components (§6.3).

        Score = clamp01(
            w1*S_err + w2*S_explore + w3*S_hammer + w4*S_burst + w5*S_persist + w6*S_spread + w7*S_cross + w8*S_dominance
            - d1*D_new - d2*D_volume - d3*D_legitUA
        ) * 100

        Special cases:
        - Strong hammering + moderate error (redirect abuse) gets synergy bonus
        - High dominance + hammering indicates coordinated network abuse

        Args:
            components: Signal components

        Returns:
            Final score (0-100)
        """
        w = self.weights

        positive_signals = (
            w.w_err * components.s_err
            + w.w_explore * components.s_explore
            + w.w_hammer * components.s_hammer
            + w.w_burst * components.s_burst
            + w.w_persist * components.s_persist
            + w.w_spread * components.s_spread
            + w.w_cross * components.s_cross
            + w.w_dominance * components.s_dominance
        )

        dampeners = (
            w.d_new * components.d_new
            + w.d_volume * components.d_volume
            + w.d_legit_ua * components.d_legit_ua
        )

        raw_score = positive_signals - dampeners

        # Special case 1: Strong hammering + error pattern (§4.2b)
        # If S_hammer > 80 and S_err > 40 (indicates redirect abuse boost was applied),
        # apply a synergy bonus since this combination is highly indicative of abuse
        if components.s_hammer > 80 and components.s_err > 40:
            synergy_bonus = 37.0
            raw_score += synergy_bonus

        # Special case 2: Dominance + hammering (network-level coordinated abuse)
        # When an entity dominates traffic AND shows hammering pattern, this indicates
        # coordinated abuse from a network (e.g., distributed scraping/hammering)
        # Lower thresholds since this is a highly specific and indicative pattern
        # Large bonus since this combination is conclusive evidence of coordinated abuse
        if components.s_dominance > 35 and components.s_hammer > 25:
            dominance_synergy = 40.0  # Increased to ensure detection
            raw_score += dominance_synergy

        return max(0.0, min(100.0, raw_score))

    def generate_explanation(self, components: SignalComponents, final_score: float) -> str:
        """
        Generate human-readable explanation (§13).

        Args:
            components: Signal components
            final_score: Final score

        Returns:
            Explanation string
        """
        reasons = []

        if components.s_err > 20:
            reasons.append(f"elevated error rate (S_err={components.s_err:.1f})")
        if components.s_explore > 20:
            reasons.append(f"abnormal exploration/scanning (S_explore={components.s_explore:.1f})")
        if components.s_hammer > 20:
            reasons.append(f"hammering/low-diversity pattern (S_hammer={components.s_hammer:.1f})")
        if components.s_burst > 20:
            reasons.append(f"rate anomaly (S_burst={components.s_burst:.1f})")
        if components.s_persist > 20:
            reasons.append(f"persistent across windows (S_persist={components.s_persist:.1f})")
        if components.s_spread > 20:
            reasons.append(f"suspicious spread (S_spread={components.s_spread:.1f})")
        if components.s_dominance > 20:
            reasons.append(f"traffic dominance (S_dominance={components.s_dominance:.1f})")

        dampener_notes = []
        if components.d_volume > 10:
            dampener_notes.append(f"low volume uncertainty (-{components.d_volume:.0f})")
        if components.d_new > 10:
            dampener_notes.append(f"new content grace (-{components.d_new:.0f})")

        if not reasons:
            reasons.append("no significant anomalies")

        explanation = "; ".join(reasons)
        if dampener_notes:
            explanation += " | Dampeners: " + ", ".join(dampener_notes)

        return explanation

    def score(self, components: SignalComponents, threshold: float) -> ScoringResult:
        """
        Compute final score and decision.

        Args:
            components: Signal components
            threshold: Score threshold for blocking

        Returns:
            ScoringResult with final score and recommendation
        """
        final_score = self.compute_final_score(components)
        explanation = self.generate_explanation(components, final_score)
        recommend_block = final_score >= threshold

        return ScoringResult(
            final_score=final_score,
            components=components,
            explanation=explanation,
            recommend_block=recommend_block,
        )


if __name__ == "__main__":
    # Test scorer
    from config import DEFAULT_CONFIG

    scorer = Scorer(DEFAULT_CONFIG.scoring_weights)

    # Test case: high error rate entity
    components = SignalComponents()
    components.s_err = scorer.compute_error_signal(
        posterior_prob=0.997, posterior_mean=0.62, theta_legit=0.08, delta=0.10
    )
    components.s_explore = scorer.compute_exploration_signal(max_robust_z=5.1)
    components.s_persist = scorer.compute_persistence_signal(num_windows=3)
    components.d_volume = scorer.compute_volume_dampener(total_requests=420, n_min=50)

    result = scorer.score(components, threshold=75.0)
    print(f"Final Score: {result.final_score:.1f}")
    print(f"Recommend Block: {result.recommend_block}")
    print(f"Explanation: {result.explanation}")
