"""
Rate anomaly tracker for burst detection (§4.3).
Maintains per-entity RateAnomalyDetector instances with bounded memory.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from botbrake.statistics import RateAnomalyDetector


class RateTracker:
    """
    Tracks request rates per entity for burst detection (§4.3).

    Maintains per-entity RateAnomalyDetector instances with:
    - EWMA baseline tracking
    - CUSUM persistent shift detection
    - Bounded memory (LRU eviction)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        cusum_k: float = 2.0,
        cusum_h: float = 5.0,
        max_entities_per_type: int = 10000,
        prune_age_minutes: int = 60,
    ):
        """
        Initialize rate tracker.

        Args:
            alpha: EWMA smoothing factor (default 0.3)
            cusum_k: CUSUM slack parameter (default 2.0)
            cusum_h: CUSUM decision threshold (default 5.0)
            max_entities_per_type: Max detectors per entity type (LRU eviction)
            prune_age_minutes: Prune entities not seen in this many minutes
        """
        self.alpha = alpha
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.max_entities_per_type = max_entities_per_type
        self.prune_age_minutes = prune_age_minutes

        # entity_type -> entity_value -> (detector, last_seen)
        self.detectors: Dict[str, Dict[str, Tuple[RateAnomalyDetector, datetime]]] = {}

        # Training mode: build baseline without emitting signals
        self.training_mode = True

    def set_training_mode(self, training: bool):
        """Set training mode (True = build baseline, no signals)."""
        self.training_mode = training

    def get_or_create_detector(
        self, entity_type: str, entity_value: str, current_time: datetime
    ) -> RateAnomalyDetector:
        """
        Get or create detector for entity.

        Args:
            entity_type: Entity type (ip, ua, cidr)
            entity_value: Entity value
            current_time: Current timestamp

        Returns:
            RateAnomalyDetector instance
        """
        if entity_type not in self.detectors:
            self.detectors[entity_type] = {}

        if entity_value not in self.detectors[entity_type]:
            # Create new detector
            detector = RateAnomalyDetector(
                alpha=self.alpha, cusum_k=self.cusum_k, cusum_h=self.cusum_h
            )
            self.detectors[entity_type][entity_value] = (detector, current_time)

            # Check if we need to evict (LRU)
            if len(self.detectors[entity_type]) > self.max_entities_per_type:
                self._evict_lru(entity_type)
        else:
            # Update last seen time
            detector, _ = self.detectors[entity_type][entity_value]
            self.detectors[entity_type][entity_value] = (detector, current_time)

        return self.detectors[entity_type][entity_value][0]

    def update_rate(
        self,
        entity_type: str,
        entity_value: str,
        window_duration_sec: float,
        request_count: int,
        current_time: datetime,
    ):
        """
        Update rate for entity.

        Args:
            entity_type: Entity type
            entity_value: Entity value
            window_duration_sec: Window duration in seconds
            request_count: Number of requests in window
            current_time: Current timestamp
        """
        if window_duration_sec <= 0:
            return

        rate = request_count / window_duration_sec
        detector = self.get_or_create_detector(entity_type, entity_value, current_time)
        detector.update(rate)

    def compute_burst_signal(
        self, entity_type: str, entity_value: str, current_rate: float, gamma: float = 0.5
    ) -> float:
        """
        Compute burst signal S_burst for entity (§4.3, §6.1).

        Args:
            entity_type: Entity type
            entity_value: Entity value
            current_rate: Current request rate (req/sec)
            gamma: Dampening exponent (default 0.5)

        Returns:
            Burst signal score (0-100), or 0 if training mode
        """
        if self.training_mode:
            return 0.0

        if entity_type not in self.detectors:
            return 0.0
        if entity_value not in self.detectors[entity_type]:
            return 0.0

        detector, _ = self.detectors[entity_type][entity_value]

        # Compute burst signal
        return detector.compute_burst_signal(current_rate, gamma)

    def get_cusum_score(self, entity_type: str, entity_value: str) -> float:
        """
        Get CUSUM score for entity.

        Args:
            entity_type: Entity type
            entity_value: Entity value

        Returns:
            CUSUM score (0-100)
        """
        if entity_type not in self.detectors:
            return 0.0
        if entity_value not in self.detectors[entity_type]:
            return 0.0

        detector, _ = self.detectors[entity_type][entity_value]
        return detector.cusum.get_score()

    def prune_old_entities(self, current_time: datetime):
        """
        Prune entities not seen recently.

        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - timedelta(minutes=self.prune_age_minutes)

        for entity_type in list(self.detectors.keys()):
            to_remove = []
            for entity_value, (_, last_seen) in self.detectors[entity_type].items():
                if last_seen < cutoff_time:
                    to_remove.append(entity_value)

            for entity_value in to_remove:
                del self.detectors[entity_type][entity_value]

    def _evict_lru(self, entity_type: str):
        """
        Evict least recently used entity for entity_type.

        Args:
            entity_type: Entity type
        """
        if entity_type not in self.detectors:
            return

        # Find LRU entity
        lru_entity = None
        lru_time = None

        for entity_value, (_, last_seen) in self.detectors[entity_type].items():
            if lru_time is None or last_seen < lru_time:
                lru_time = last_seen
                lru_entity = entity_value

        if lru_entity:
            del self.detectors[entity_type][lru_entity]

    def get_stats(self) -> Dict:
        """
        Get statistics about tracked entities.

        Returns:
            Dict with entity counts per type
        """
        return {
            entity_type: len(entities) for entity_type, entities in self.detectors.items()
        }


if __name__ == "__main__":
    # Test rate tracker
    from datetime import datetime

    tracker = RateTracker()
    tracker.set_training_mode(False)  # Enable detection

    ts = datetime.now()

    # Simulate normal traffic
    for i in range(10):
        tracker.update_rate("ip", "192.168.1.1", 60.0, 100, ts)

    # Simulate burst
    burst_rate = 1000 / 60.0  # 1000 req in 60s
    signal = tracker.compute_burst_signal("ip", "192.168.1.1", burst_rate)

    print(f"Burst signal: {signal:.1f}")
    print(f"Stats: {tracker.get_stats()}")