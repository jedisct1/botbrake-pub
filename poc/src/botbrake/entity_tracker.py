"""
Entity tracker for per-entity counters and metrics (§3).
Tracks IP, UA, path, referrer, and CIDR behavior across time windows.
"""

from datetime import datetime
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from botbrake.cidr_utils import extract_cidr_prefix


@dataclass
class EntityMetrics:
    """
    Per-entity metrics for a time window (§3).
    Tracks requests, errors, paths, and other signals.
    """

    entity_type: str  # 'ip', 'ua', 'path', 'ref'
    entity_value: str
    window_start: datetime
    window_end: datetime

    # Basic counters
    total_requests: int = 0
    error_count: int = 0  # 4xx + 5xx

    # Status distribution
    status_2xx: int = 0
    status_3xx: int = 0
    status_4xx: int = 0
    status_5xx: int = 0

    # Cardinality tracking
    unique_paths: Set[str] = field(default_factory=set)
    unique_refs: Set[str] = field(default_factory=set)
    unique_uas: Set[str] = field(default_factory=set)  # For IP entity
    unique_ips: Set[str] = field(default_factory=set)  # For UA entity

    # Path frequency tracking (for hammering detection)
    path_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Method distribution
    methods: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Prefix fanout (tracked separately per depth)
    prefixes_by_depth: Dict[int, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # Depth distribution tracking (§2.3)
    # Maps depth -> count of requests reaching that depth
    depth_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # First/last seen in this window
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    def add_request(
        self,
        path: str,
        status_class: str,
        method: str,
        timestamp: datetime,
        ref: Optional[str] = None,
        ua: Optional[str] = None,
        ip: Optional[str] = None,
    ):
        """Add a request to this entity's metrics."""
        self.total_requests += 1

        # Update status counts
        if status_class == "2xx":
            self.status_2xx += 1
        elif status_class == "3xx":
            self.status_3xx += 1
        elif status_class == "4xx":
            self.status_4xx += 1
            self.error_count += 1
        elif status_class == "5xx":
            self.status_5xx += 1
            self.error_count += 1

        # Track unique values
        self.unique_paths.add(path)
        self.path_counts[path] += 1  # Track frequency for hammering detection

        if ref:
            self.unique_refs.add(ref)
        if ua and self.entity_type == "ip":
            self.unique_uas.add(ua)
        if ip and self.entity_type == "ua":
            self.unique_ips.add(ip)

        # Track methods
        self.methods[method] += 1

        # Update timestamps
        if self.first_seen is None or timestamp < self.first_seen:
            self.first_seen = timestamp
        if self.last_seen is None or timestamp > self.last_seen:
            self.last_seen = timestamp

    def add_prefix(self, depth: int, prefix: str):
        """Track a prefix at a specific depth."""
        self.prefixes_by_depth[depth].add(prefix)

    def add_depth(self, depth: int):
        """
        Track depth reached by a request (§2.3).

        Args:
            depth: Deepest prefix depth for this request
        """
        self.depth_counts[depth] += 1

    def compute_depth_score(self) -> float:
        """
        Compute depth score: weighted average of depths reached (§2.3).

        depth_score = Σ(count_at_depth_d × d) / total_requests

        Returns:
            Depth score (higher = deeper navigation)
        """
        if self.total_requests == 0 or not self.depth_counts:
            return 0.0

        weighted_sum = sum(depth * count for depth, count in self.depth_counts.items())
        return weighted_sum / self.total_requests

    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    def explore_ratio(self) -> float:
        """Calculate exploration ratio: unique_paths / total_requests."""
        if self.total_requests == 0:
            return 0.0
        return len(self.unique_paths) / self.total_requests

    def prefix_fanout(self, depth: int) -> float:
        """Calculate prefix fanout at depth: unique_prefixes@depth / total_requests."""
        if self.total_requests == 0:
            return 0.0
        return len(self.prefixes_by_depth.get(depth, set())) / self.total_requests

    def ua_diversity(self) -> int:
        """Number of unique UAs (for IP entity)."""
        return len(self.unique_uas)

    def ip_diversity(self) -> int:
        """Number of unique IPs (for UA entity)."""
        return len(self.unique_ips)

    def top_path_ratio(self) -> float:
        """
        Calculate ratio of requests to most frequent path (§4.2b).
        Used for hammering detection.

        Returns:
            Ratio (0-1) of requests to the most hit path
        """
        if self.total_requests == 0 or not self.path_counts:
            return 0.0

        max_path_count = max(self.path_counts.values())
        return max_path_count / self.total_requests

    def redirect_rate(self) -> float:
        """
        Calculate 3xx redirect rate (§4.2b).

        Returns:
            Ratio (0-1) of 3xx responses
        """
        if self.total_requests == 0:
            return 0.0
        return self.status_3xx / self.total_requests

    def path_concentration(self) -> float:
        """
        Calculate path concentration: 1 - explore_ratio (§4.2b).
        High concentration = hammering.

        Returns:
            Concentration (0-1)
        """
        return 1.0 - self.explore_ratio()

    def to_dict(self) -> dict:
        """Convert to dictionary for output."""
        return {
            "entity_type": self.entity_type,
            "entity_value": self.entity_value,
            "window": f"{self.window_start} - {self.window_end}",
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_rate(),
            "explore_ratio": self.explore_ratio(),
            "unique_paths": len(self.unique_paths),
            "unique_uas": len(self.unique_uas),
            "unique_ips": len(self.unique_ips),
            "status_distribution": {
                "2xx": self.status_2xx,
                "3xx": self.status_3xx,
                "4xx": self.status_4xx,
                "5xx": self.status_5xx,
            },
        }


class EntityTracker:
    """
    Tracks metrics across all entities for time windows (§3).
    """

    def __init__(self):
        """Initialize entity tracker."""
        # entity_type -> entity_value -> window_id -> EntityMetrics
        self.metrics: Dict[str, Dict[str, Dict[str, EntityMetrics]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def get_or_create_metrics(
        self,
        entity_type: str,
        entity_value: str,
        window_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> EntityMetrics:
        """Get or create metrics for an entity in a window."""
        if window_id not in self.metrics[entity_type][entity_value]:
            self.metrics[entity_type][entity_value][window_id] = EntityMetrics(
                entity_type=entity_type,
                entity_value=entity_value,
                window_start=window_start,
                window_end=window_end,
            )
        return self.metrics[entity_type][entity_value][window_id]

    def get_metrics(
        self, entity_type: str, entity_value: str, window_id: str
    ) -> Optional[EntityMetrics]:
        """Get metrics for an entity in a window."""
        return self.metrics.get(entity_type, {}).get(entity_value, {}).get(window_id)

    def get_all_entities(self, entity_type: str) -> List[str]:
        """Get all entity values for a given type."""
        return list(self.metrics.get(entity_type, {}).keys())

    def get_entity_windows(self, entity_type: str, entity_value: str) -> List[str]:
        """Get all window IDs for a specific entity."""
        return list(self.metrics.get(entity_type, {}).get(entity_value, {}).keys())

    def get_all_metrics_for_window(self, entity_type: str, window_id: str) -> List[EntityMetrics]:
        """Get all entity metrics for a specific window."""
        result = []
        for entity_value in self.metrics.get(entity_type, {}).keys():
            metrics = self.get_metrics(entity_type, entity_value, window_id)
            if metrics:
                result.append(metrics)
        return result

    def aggregate_cidr_metrics(
        self, window_id: str, window_start: datetime, window_end: datetime, ipv4_prefix: int = 24, ipv6_prefix: int = 48
    ) -> Dict[str, EntityMetrics]:
        """
        Aggregate IP metrics by CIDR prefix.

        Args:
            window_id: Window identifier
            window_start: Window start time
            window_end: Window end time
            ipv4_prefix: IPv4 prefix length (default /24)
            ipv6_prefix: IPv6 prefix length (default /48)

        Returns:
            Dictionary mapping CIDR prefix to aggregated metrics
        """
        cidr_metrics = {}

        # Iterate through all IPs for this window
        for ip_value in self.metrics.get("ip", {}).keys():
            metrics = self.get_metrics("ip", ip_value, window_id)
            if not metrics:
                continue

            # Extract CIDR prefix
            cidr_prefix = extract_cidr_prefix(ip_value, ipv4_prefix, ipv6_prefix)

            # Create or get CIDR metrics
            if cidr_prefix not in cidr_metrics:
                cidr_metrics[cidr_prefix] = EntityMetrics(
                    entity_type="cidr",
                    entity_value=cidr_prefix,
                    window_start=window_start,
                    window_end=window_end,
                )

            # Aggregate metrics
            cidr_agg = cidr_metrics[cidr_prefix]
            cidr_agg.total_requests += metrics.total_requests
            cidr_agg.error_count += metrics.error_count
            cidr_agg.status_2xx += metrics.status_2xx
            cidr_agg.status_3xx += metrics.status_3xx
            cidr_agg.status_4xx += metrics.status_4xx
            cidr_agg.status_5xx += metrics.status_5xx

            # Merge sets
            cidr_agg.unique_paths.update(metrics.unique_paths)
            cidr_agg.unique_refs.update(metrics.unique_refs)
            cidr_agg.unique_uas.update(metrics.unique_uas)
            cidr_agg.unique_ips.add(ip_value)  # Track which IPs are in this CIDR

            # Merge path counts
            for path, count in metrics.path_counts.items():
                cidr_agg.path_counts[path] += count

            # Merge methods
            for method, count in metrics.methods.items():
                cidr_agg.methods[method] += count

            # Update timestamps
            if metrics.first_seen:
                if cidr_agg.first_seen is None or metrics.first_seen < cidr_agg.first_seen:
                    cidr_agg.first_seen = metrics.first_seen
            if metrics.last_seen:
                if cidr_agg.last_seen is None or metrics.last_seen > cidr_agg.last_seen:
                    cidr_agg.last_seen = metrics.last_seen

        return cidr_metrics


class PersistenceTracker:
    """
    Tracks signal persistence across windows (§4, §6.1).
    Used to determine if abnormal behavior is sustained.
    """

    def __init__(self):
        """Initialize persistence tracker."""
        # entity_type -> entity_value -> list of (window_id, had_signal)
        self.history: Dict[str, Dict[str, List[tuple[str, bool]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record_signal(self, entity_type: str, entity_value: str, window_id: str, has_signal: bool):
        """Record whether an entity had a signal in a window."""
        self.history[entity_type][entity_value].append((window_id, has_signal))

    def get_consecutive_windows(self, entity_type: str, entity_value: str) -> int:
        """Count consecutive windows with signals."""
        history = self.history.get(entity_type, {}).get(entity_value, [])
        if not history:
            return 0

        # Count from end
        count = 0
        for _, has_signal in reversed(history):
            if has_signal:
                count += 1
            else:
                break
        return count

    def get_total_windows_with_signal(self, entity_type: str, entity_value: str) -> int:
        """Count total windows with signals."""
        history = self.history.get(entity_type, {}).get(entity_value, [])
        return sum(1 for _, has_signal in history if has_signal)


if __name__ == "__main__":
    # Test entity tracker
    tracker = EntityTracker()
    ts = datetime.now()

    metrics = tracker.get_or_create_metrics("ip", "192.168.1.1", "window_1m_0", ts, ts)
    metrics.add_request("/path1", "4xx", "GET", ts, ua="Mozilla/5.0")
    metrics.add_request("/path2", "2xx", "GET", ts, ua="Mozilla/5.0")
    metrics.add_request("/path1", "4xx", "POST", ts, ua="curl/8.0")

    print(f"Error rate: {metrics.error_rate():.2f}")
    print(f"Explore ratio: {metrics.explore_ratio():.2f}")
    print(f"UA diversity: {metrics.ua_diversity()}")
