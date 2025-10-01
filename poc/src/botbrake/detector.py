"""
Main detection engine (§5, §9).
Implements entity-specific detectors and multi-signal consensus.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from botbrake.config import DetectionConfig
from botbrake.entity_tracker import EntityMetrics
from botbrake.statistics import BetaBinomialModel, BaselineStatistics
from botbrake.scorer import Scorer, SignalComponents
from botbrake.prefix_trie import PrefixTrie
from botbrake.rate_tracker import RateTracker


@dataclass
class DetectionDecision:
    """Detection decision output (§13)."""

    when: str  # Timestamp
    entity_type: str  # ip, ua, path, ref
    entity_value: str
    score: float  # 0-100
    recommended_action: str  # "block", "monitor"
    duration_sec: Optional[int]  # Block duration
    signals: Dict  # Detailed signal breakdown
    dampeners: Dict  # Dampener details
    explanation: str


class Detector:
    """
    Main abuse detector (§5, §9).
    Implements multi-signal consensus and entity-specific logic.
    """

    def __init__(
        self,
        config: DetectionConfig,
        baseline: BaselineStatistics,
        rate_tracker: Optional[RateTracker] = None,
    ):
        """
        Initialize detector.

        Args:
            config: Detection configuration
            baseline: Learned baseline statistics
            rate_tracker: Optional rate tracker for burst detection (§4.3)
        """
        self.config = config
        self.baseline = baseline
        self.model = BetaBinomialModel(baseline.beta_prior)
        self.scorer = Scorer(config.scoring_weights)
        self.rate_tracker = rate_tracker

    def analyze_entity(
        self,
        metrics: EntityMetrics,
        prefix_trie: PrefixTrie,
        persistence_count: int,
        window_id: str,
        total_window_requests: int = 0,
        window_duration_sec: float = 300.0,
    ) -> Optional[DetectionDecision]:
        """
        Analyze an entity and generate decision if needed (§9).

        Args:
            metrics: Entity metrics for the window
            prefix_trie: Global prefix trie for context
            persistence_count: Number of consecutive windows with signals
            window_id: Window identifier
            total_window_requests: Total requests in window (for dominance)
            window_duration_sec: Window duration in seconds (for burst detection)

        Returns:
            DetectionDecision if entity should be blocked, else None
        """
        # Extract entity info
        entity_type = metrics.entity_type
        entity_value = metrics.entity_value

        # Get threshold for this entity type
        if entity_type == "ip":
            threshold = self.config.score_thresholds.ip
        elif entity_type == "ua":
            threshold = self.config.score_thresholds.ua
        elif entity_type == "path":
            threshold = self.config.score_thresholds.path
        elif entity_type == "cidr":
            threshold = self.config.score_thresholds.cidr
        else:
            threshold = 75.0  # Default

        # Check minimum evidence
        n_min = self.config.thresholds.n_min.get(
            self.config.windows.windows_sec[0], 50  # Use smallest window for now
        )

        # Compute signal components
        components = self._compute_signals(
            metrics, prefix_trie, persistence_count, n_min, total_window_requests, window_duration_sec
        )

        # Score entity
        result = self.scorer.score(components, threshold)

        # Check consensus requirements
        if not self._check_consensus(entity_type, components, metrics):
            return None  # Consensus not met, don't block

        # Only return decision if recommended
        if not result.recommend_block:
            return None

        # Map score to duration
        duration_sec = self._map_score_to_duration(result.final_score, entity_type)

        # Create decision
        return DetectionDecision(
            when=str(metrics.window_end),
            entity_type=entity_type,
            entity_value=entity_value,
            score=result.final_score,
            recommended_action="block",
            duration_sec=duration_sec,
            signals=self._signals_to_dict(metrics, components),
            dampeners=self._dampeners_to_dict(components),
            explanation=result.explanation,
        )

    def _compute_signals(
        self,
        metrics: EntityMetrics,
        prefix_trie: PrefixTrie,
        persistence_count: int,
        n_min: int,
        total_window_requests: int = 0,
        window_duration_sec: float = 300.0,
    ) -> SignalComponents:
        """Compute all signal components."""
        components = SignalComponents()

        # 1. Error signal (S_err)
        if self.baseline.global_error_rate is not None:
            theta_legit = self.baseline.global_error_rate
            delta = self.config.thresholds.delta_error_pp

            posterior_prob = self.model.posterior_probability_exceeds(
                metrics.error_count, metrics.total_requests, theta_legit + delta
            )
            posterior_mean = self.model.posterior_mean(metrics.error_count, metrics.total_requests)

            components.s_err = self.scorer.compute_error_signal(
                posterior_prob, posterior_mean, theta_legit, delta
            )

        # 2. Exploration signal (S_explore) - HIGH diversity scanning
        # NOTE: Only positive z-scores (above baseline) indicate scanning
        # Negative z-scores (below baseline) indicate hammering, which is captured by s_hammer
        explore_rz = self.baseline.get_explore_ratio_rz(metrics.explore_ratio())

        # Compute prefix fanout z-scores at multiple depths
        prefix_rz_values = []
        for depth in self.config.prefix.depths:
            fanout = metrics.prefix_fanout(depth)
            if fanout > 0:
                rz = self.baseline.get_prefix_fanout_rz(depth, fanout)
                # Only use positive z-scores for exploration (above baseline)
                if rz > 0:
                    prefix_rz_values.append(rz)

        # Compute depth score z-score (§2.3)
        # Shallow navigation (negative z-score) indicates scanning
        depth_score = metrics.compute_depth_score()
        depth_rz = self.baseline.get_depth_score_rz(depth_score)

        # Use maximum z-score, but only if positive (above baseline)
        all_rz = ([explore_rz] if explore_rz > 0 else []) + prefix_rz_values

        # Add depth score if shallow (negative z-score < -3.0 indicates shallow scanning)
        if depth_rz < -3.0:
            all_rz.append(abs(depth_rz))

        max_rz = max(all_rz) if all_rz else 0.0

        components.s_explore = self.scorer.compute_exploration_signal(max_rz)

        # 3. Dominance signal (S_dominance) - Network-level traffic dominance
        traffic_ratio = 0.0
        if total_window_requests > 0:
            traffic_ratio = metrics.total_requests / total_window_requests
            components.s_dominance = self.scorer.compute_dominance_signal(
                metrics.total_requests,
                total_window_requests,
                dominance_threshold=self.config.cidr.dominance_threshold,
            )
        else:
            components.s_dominance = 0.0

        # 4. Hammering signal (S_hammer) - LOW diversity concentration
        top_path_ratio = metrics.top_path_ratio()
        explore_ratio = metrics.explore_ratio()

        # Use lower threshold for dominant networks (more sensitive detection)
        # Check raw traffic ratio, not signal score, to avoid chicken-and-egg problem
        is_dominant = traffic_ratio >= self.config.cidr.dominance_threshold
        if is_dominant and metrics.total_requests >= self.config.thresholds.n_hammer_min:
            # For dominant networks, detect hammering based on LOW PATH DIVERSITY
            # rather than single-path concentration
            # Reason: Abuse may spread requests across a few URLs (e.g., 4 URLs)
            # to evade simple top-path detection

            # Use path concentration (1 - explore_ratio) as hammering indicator
            # Very low explore_ratio (<1%) indicates hammering even if spread across a few paths
            path_concentration = 1.0 - explore_ratio

            # For dominant networks: concentration > 99% is suspicious
            # Linear scale: 99% → 0 score, 99.99% → 100 score
            concentration_threshold = 0.99
            if path_concentration >= concentration_threshold:
                normalized = (path_concentration - concentration_threshold) / (1.0 - concentration_threshold)
                components.s_hammer = 100.0 * max(0.0, min(1.0, normalized))
            else:
                # Fall back to top-path ratio if concentration not extreme
                hammer_min = self.config.cidr.dominant_network_hammer_min
                hammer_max = self.config.cidr.dominant_network_hammer_max
                adjusted_ratio = (top_path_ratio - hammer_min) / (hammer_max - hammer_min)
                adjusted_ratio = max(0.0, min(1.0, adjusted_ratio))
                components.s_hammer = 100.0 * adjusted_ratio
        else:
            # Normal hammering detection
            components.s_hammer = self.scorer.compute_hammer_signal(
                top_path_ratio, metrics.total_requests, n_hammer_min=self.config.thresholds.n_hammer_min
            )

        # Apply 3xx redirect abuse boost if applicable (§4.2b)
        redirect_rate = metrics.redirect_rate()
        if redirect_rate >= 0.95 and top_path_ratio >= 0.85:
            # Heavy redirects + hammering = treat as error-equivalent
            # This indicates redirect abuse/hammering which is malicious
            redirect_abuse_factor = min(1.0, redirect_rate * top_path_ratio)
            # Boost error signal significantly - this is abuse even without 4xx/5xx
            boost = 50.0 * redirect_abuse_factor  # Increased from 30.0 to 50.0
            components.s_err = min(100.0, components.s_err + boost)

        # 5. Burst signal (S_burst) - Rate anomaly (EWMA/CUSUM) (§4.3)
        if self.rate_tracker:
            # Compute current rate
            current_rate = metrics.total_requests / window_duration_sec if window_duration_sec > 0 else 0.0

            # Compute burst signal using rate tracker
            components.s_burst = self.rate_tracker.compute_burst_signal(
                metrics.entity_type, metrics.entity_value, current_rate
            )
        else:
            components.s_burst = 0.0  # No rate tracker available

        # 6. Persistence signal (S_persist)
        components.s_persist = self.scorer.compute_persistence_signal(persistence_count)

        # 7. Spread signal (S_spread)
        if metrics.entity_type == "ua":
            # UA spread across IPs
            components.s_spread = self.scorer.compute_spread_signal(
                metrics.ip_diversity(), velocity=0.0  # Simplified
            )
        elif metrics.entity_type == "ip":
            # IP spread across UAs (spoofing)
            if metrics.ua_diversity() >= 5:
                components.s_spread = min(100.0, metrics.ua_diversity() * 10.0)
        else:
            components.s_spread = 0.0

        # 8. Cross signal (S_cross) - simplified for MVP
        components.s_cross = 0.0

        # Dampeners
        components.d_volume = self.scorer.compute_volume_dampener(metrics.total_requests, n_min)

        # New content dampener (§4.4)
        components.d_new = self._compute_new_content_dampener(metrics, prefix_trie)

        components.d_legit_ua = 0.0  # No hardcoded UA allowlist per requirements

        return components

    def _compute_new_content_dampener(
        self, metrics: EntityMetrics, prefix_trie: PrefixTrie
    ) -> float:
        """
        Compute new content dampener (§4.4).

        Checks if entity's paths hit new prefixes (TOFS < grace_period).
        Returns dampener value 0-30 based on fraction of traffic to new content.

        Args:
            metrics: Entity metrics
            prefix_trie: Global prefix trie

        Returns:
            Dampener value (0-30)
        """
        if metrics.total_requests == 0:
            return 0.0

        # Configuration from §4.4 algorithm spec
        grace_period_min = 90
        min_unique_ips = 100
        error_rate_threshold = 0.20

        # Count requests to new content
        new_content_requests = 0
        current_time = metrics.window_end

        # Check each unique path
        for path in metrics.unique_paths:
            # Check prefix at depth 6 (balance between specificity and coverage)
            prefix = prefix_trie.extract_prefix(path, 6)

            # Get TOFS for this prefix
            tofs = prefix_trie.get_tofs(prefix, 6)
            if not tofs:
                continue

            # Check if prefix is new
            age_min = (current_time - tofs).total_seconds() / 60
            if age_min > grace_period_min:
                continue

            # Check unique IPs and error rate
            unique_ips = prefix_trie.get_unique_ip_count(prefix, 6)
            error_rate = prefix_trie.get_error_rate(prefix, 6)

            # Is this new content (recent, diverse traffic, normal errors)?
            if unique_ips >= min_unique_ips and error_rate < error_rate_threshold:
                # Count requests to this path
                new_content_requests += metrics.path_counts.get(path, 0)

        # Compute fraction
        new_content_fraction = new_content_requests / metrics.total_requests

        # Dampener scales with new content fraction (max 30 points per §6.2)
        return 30.0 * new_content_fraction

    def _check_consensus(
        self, entity_type: str, components: SignalComponents, metrics: EntityMetrics
    ) -> bool:
        """
        Check multi-signal consensus requirements (§5, §8).

        Updated to include hammering and rate signals (§5.2).

        Args:
            entity_type: Type of entity
            components: Signal components
            metrics: Entity metrics

        Returns:
            True if consensus requirements met
        """
        # Count significant signals (threshold: 20/100)
        signal_count = 0

        if components.s_err > 20:
            signal_count += 1
        if components.s_explore > 20:  # Scanning (HIGH diversity)
            signal_count += 1
        if components.s_hammer > 20:  # Hammering (LOW diversity)
            signal_count += 1
        if components.s_burst > 20:  # Rate anomaly
            signal_count += 1
        if components.s_persist > 20:
            signal_count += 1
        if components.s_spread > 20:
            signal_count += 1
        if components.s_dominance > 20:  # Traffic dominance
            signal_count += 1

        # Path blocks are safest - allow with fewer signals
        if entity_type == "path":
            return signal_count >= 1  # Path blocks safer, need only 1 strong signal

        # IP/UA blocks require multi-signal consensus (§5.2)
        # Exception 1: hammering + rate spike can block even without errors (§4.3)
        if components.s_hammer > 60 and components.s_burst > 60:
            return True  # DDoS pattern exception

        # Exception 2: dominance + hammering indicates coordinated network abuse
        # Lower thresholds than other exceptions since this is highly specific pattern
        if components.s_dominance > 35 and components.s_hammer > 25:
            return True  # Network dominance exception

        min_signals = (
            self.config.min_signals_for_ip_block
            if entity_type == "ip"
            else self.config.min_signals_for_ua_block
        )

        return signal_count >= min_signals

    def _map_score_to_duration(self, score: float, entity_type: str) -> int:
        """
        Map score to block duration in seconds (§7).

        Piecewise rule:
        - Score < 60: No block (0 sec)
        - 60-74: 15-60 min for path/ref
        - 75-89: 2^((score-70)/10) * 10 min for IP/UA, clipped to 2-240 min
        - ≥90: 2^((score-80)/7) * 30 min for IP/UA/CIDR, clipped to 6-24h

        Args:
            score: Final score (0-100)
            entity_type: Entity type

        Returns:
            Duration in seconds
        """
        if score < 60:
            return 0

        if score < 75:
            # Soft block: 15-60 min
            if entity_type in ("path", "ref"):
                minutes = 15 + (score - 60) * 3  # Linear ramp 15-60
                return int(min(60, max(15, minutes)) * 60)
            else:
                return 0  # Don't block IP/UA at low scores

        if score < 90:
            # Medium block: IP/UA
            if entity_type in ("ip", "ua"):
                minutes = (2 ** ((score - 70) / 10.0)) * 10
                minutes = max(2, min(240, minutes))  # Clip 2-240 min
                return int(minutes * 60)
            else:
                return 60 * 60  # 1 hour default

        # High score: aggressive block
        minutes = (2 ** ((score - 80) / 7.0)) * 30
        minutes = max(6 * 60, min(24 * 60, minutes))  # Clip 6-24 hours
        return int(minutes * 60)

    def _signals_to_dict(self, metrics: EntityMetrics, components: SignalComponents) -> Dict:
        """Convert signals to dictionary for output."""
        return {
            "error_posterior": {
                "error_rate": metrics.error_rate(),
                "error_count": metrics.error_count,
                "total": metrics.total_requests,
                "score": components.s_err,
            },
            "exploration": {
                "explore_ratio": metrics.explore_ratio(),
                "unique_paths": len(metrics.unique_paths),
                "score": components.s_explore,
            },
            "hammering": {
                "top_path_ratio": metrics.top_path_ratio(),
                "path_concentration": metrics.path_concentration(),
                "redirect_rate": metrics.redirect_rate(),
                "score": components.s_hammer,
            },
            "dominance": {
                "score": components.s_dominance,
            },
            "burst": {"score": components.s_burst},
            "persistence": {"score": components.s_persist},
            "spread": {
                "unique_ips": len(metrics.unique_ips),
                "unique_uas": len(metrics.unique_uas),
                "score": components.s_spread,
            },
        }

    def _dampeners_to_dict(self, components: SignalComponents) -> Dict:
        """Convert dampeners to dictionary for output."""
        return {
            "volume": components.d_volume,
            "new_content": components.d_new,
            "legit_ua": components.d_legit_ua,
        }


if __name__ == "__main__":
    # Test detector
    from datetime import datetime
    from botbrake.config import DEFAULT_CONFIG
    from botbrake.statistics import BetaBinomialPrior

    # Create baseline
    baseline = BaselineStatistics()
    baseline.set_global_error_rate(0.08)
    baseline.set_beta_prior(BetaBinomialPrior.from_error_rates([0.05, 0.08, 0.06]))
    baseline.add_explore_ratio(0.1)
    baseline.add_explore_ratio(0.12)

    detector = Detector(DEFAULT_CONFIG, baseline)

    # Create test metrics (simulating abusive IP)
    ts = datetime.now()
    metrics = EntityMetrics(
        entity_type="ip", entity_value="154.249.137.8", window_start=ts, window_end=ts
    )

    # Simulate high error rate, high exploration
    for i in range(100):
        metrics.add_request(f"/path{i}", "3xx", "GET", ts)

    trie = PrefixTrie([2, 4, 6, 8])

    decision = detector.analyze_entity(metrics, trie, persistence_count=3, window_id="test")
    if decision:
        print(f"Decision: Block {decision.entity_value}")
        print(f"Score: {decision.score:.1f}")
        print(f"Duration: {decision.duration_sec} seconds")
        print(f"Explanation: {decision.explanation}")
    else:
        print("No block decision")
