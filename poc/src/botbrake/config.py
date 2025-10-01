"""
Configuration module for BotBrake abuse detection system.
Based on algorithm.md §17 - Minimal Config (starter values).
"""

from dataclasses import dataclass
from typing import List


@dataclass
class WindowConfig:
    """Time window configuration."""

    windows_sec: List[int] = None  # [60, 300, 3600]  # 1m, 5m, 1h

    def __post_init__(self):
        if self.windows_sec is None:
            self.windows_sec = [60, 300, 3600]  # 1m, 5m, 1h


@dataclass
class ThresholdConfig:
    """Minimum evidence thresholds per window (§10)."""

    # Minimum request count per window for decisions
    n_min: dict = None

    # Minimum requests for hammering detection (§4.2b)
    n_hammer_min: int = 500

    # Error rate tolerance margin (percentage points)
    delta_error_pp: float = 0.10

    # Robust z-score threshold for exploration metrics
    scan_z_threshold: float = 4.0

    # Rate anomaly tail probability threshold
    rate_tail_p_max: float = 1e-6

    def __post_init__(self):
        if self.n_min is None:
            # Conservative: require more evidence for shorter windows
            self.n_min = {
                60: 50,  # 1m: 50 requests minimum
                300: 200,  # 5m: 200 requests minimum
                3600: 500,  # 1h: 500 requests minimum
            }


@dataclass
class NewContentConfig:
    """New content protection to avoid false positives (§4.4)."""

    grace_minutes: int = 90
    min_unique_ips: int = 100


@dataclass
class CIDRConfig:
    """CIDR network blocking configuration (§11)."""

    ipv4_test_prefix: int = 24
    ipv6_test_prefix: int = 48  # Use /48 for IPv6 (more appropriate for network aggregation)

    # Dominance detection thresholds
    dominance_threshold: float = 0.30  # 30% of traffic from single network triggers dominance signal

    # Lower hammering threshold and range for dominant networks
    dominant_network_hammer_min: float = 0.30  # Start detecting at 30% concentration (vs 50% for normal)
    dominant_network_hammer_max: float = 0.70  # 70% is considered max/extreme hammering (vs 90% for normal)


@dataclass
class ScoreThresholds:
    """Score thresholds per entity type for blocking decisions (§7)."""

    path: float = 60.0
    ip: float = 75.0
    ua: float = 75.0
    cidr: float = 50.0  # Lower threshold for CIDR blocks with dominance+hammering (very specific pattern)


@dataclass
class ScoringWeights:
    """Component signal weights for scoring function (§6.3, updated)."""

    # Signal weights (should sum to ~1.0)
    w_err: float = 0.28  # Error propensity
    w_explore: float = 0.18  # Exploration/scanning (HIGH diversity)
    w_hammer: float = 0.18  # Hammering/concentration (LOW diversity)
    w_burst: float = 0.12  # Rate anomaly (EWMA/CUSUM)
    w_persist: float = 0.10  # Persistence across windows
    w_spread: float = 0.05  # Spread across entities
    w_cross: float = 0.03  # Cross-entity corroboration
    w_dominance: float = 0.06  # Traffic dominance (network abuse)

    # Dampener weights
    d_new: float = 1.0  # New content leniency
    d_volume: float = 1.0  # Low volume uncertainty
    d_legit_ua: float = 1.0  # Known good crawler


@dataclass
class PrefixConfig:
    """URICrypt prefix analysis configuration (§2)."""

    # Depths at which to extract and analyze prefixes
    depths: List[int] = None

    def __post_init__(self):
        if self.depths is None:
            self.depths = [2, 4, 6, 8, 16, 32, 64]


@dataclass
class BayesianConfig:
    """Bayesian statistical model configuration (§4.1, Appendix A)."""

    # Initial weak prior for Beta-Binomial
    alpha_0: float = 1.0
    beta_0: float = 1.0

    # Confidence threshold for posterior decisions
    posterior_confidence: float = 0.99  # P(θ_E > θ_legit + δ) > 0.99


@dataclass
class DetectionConfig:
    """Main detection system configuration."""

    windows: WindowConfig = None
    thresholds: ThresholdConfig = None
    new_content: NewContentConfig = None
    cidr: CIDRConfig = None
    score_thresholds: ScoreThresholds = None
    scoring_weights: ScoringWeights = None
    prefix: PrefixConfig = None
    bayesian: BayesianConfig = None

    # Multi-signal consensus: minimum number of independent signals required
    min_signals_for_ip_block: int = 2
    min_signals_for_ua_block: int = 2

    # Persistence: minimum consecutive/total windows showing signals
    min_windows_for_persistence: int = 2

    def __post_init__(self):
        if self.windows is None:
            self.windows = WindowConfig()
        if self.thresholds is None:
            self.thresholds = ThresholdConfig()
        if self.new_content is None:
            self.new_content = NewContentConfig()
        if self.cidr is None:
            self.cidr = CIDRConfig()
        if self.score_thresholds is None:
            self.score_thresholds = ScoreThresholds()
        if self.scoring_weights is None:
            self.scoring_weights = ScoringWeights()
        if self.prefix is None:
            self.prefix = PrefixConfig()
        if self.bayesian is None:
            self.bayesian = BayesianConfig()


# Default configuration instance
DEFAULT_CONFIG = DetectionConfig()
