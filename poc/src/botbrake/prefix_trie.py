"""
Patricia Trie for URICrypt prefix analysis (§2).
Exploits prefix-preserving property of URICrypt encryption.
"""

from datetime import datetime
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TrieNode:
    """
    Node in the prefix trie (§2.1).
    Tracks metrics at a specific prefix depth.
    """

    prefix: str
    depth: int

    # Metrics
    count: int = 0  # Hit count
    unique_ips: Set[str] = field(default_factory=set)
    unique_uas: Set[str] = field(default_factory=set)
    status_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tofs: Optional[datetime] = None  # Time of first seen

    # Children (for full trie structure if needed)
    children: Dict[str, "TrieNode"] = field(default_factory=dict)

    def update(self, ip: str, ua: Optional[str], status_class: str, timestamp: datetime):
        """Update node metrics with a new request."""
        self.count += 1
        self.unique_ips.add(ip)
        if ua:
            self.unique_uas.add(ua)
        self.status_distribution[status_class] += 1

        # Update TOFS
        if self.tofs is None or timestamp < self.tofs:
            self.tofs = timestamp

    def error_rate(self) -> float:
        """Calculate error rate (4xx + 5xx) / total."""
        if self.count == 0:
            return 0.0
        errors = self.status_distribution.get("4xx", 0) + self.status_distribution.get("5xx", 0)
        return errors / self.count

    def unique_ip_count(self) -> int:
        """Number of unique IPs that hit this prefix."""
        return len(self.unique_ips)

    def unique_ua_count(self) -> int:
        """Number of unique UAs that hit this prefix."""
        return len(self.unique_uas)


class PrefixTrie:
    """
    URICrypt-aware prefix trie (§2).
    Maintains prefix analytics at multiple depths.
    """

    def __init__(self, depths: List[int]):
        """
        Initialize trie with specified analysis depths.

        Args:
            depths: List of character depths to analyze (e.g., [2, 4, 6, 8, 16, 32])
        """
        self.depths = sorted(depths)
        # Prefix -> TrieNode mapping per depth
        self.nodes_by_depth: Dict[int, Dict[str, TrieNode]] = {depth: {} for depth in self.depths}

    def add_path(
        self, path: str, ip: str, ua: Optional[str], status_class: str, timestamp: datetime
    ):
        """
        Add a path to the trie, updating all relevant depth nodes (§2.1).

        Args:
            path: Encrypted path (URICrypt ciphertext)
            ip: Client IP
            ua: User agent (or None)
            status_class: Status class (2xx, 3xx, 4xx, 5xx)
            timestamp: Request timestamp
        """
        for depth in self.depths:
            if len(path) < depth:
                # Path shorter than depth, skip
                continue

            prefix = path[:depth]

            # Get or create node at this depth
            if prefix not in self.nodes_by_depth[depth]:
                self.nodes_by_depth[depth][prefix] = TrieNode(prefix=prefix, depth=depth)

            # Update node
            self.nodes_by_depth[depth][prefix].update(ip, ua, status_class, timestamp)

    def get_node(self, prefix: str, depth: int) -> Optional[TrieNode]:
        """Get node for a specific prefix at a specific depth."""
        return self.nodes_by_depth.get(depth, {}).get(prefix)

    def get_prefixes_at_depth(self, depth: int) -> List[str]:
        """Get all prefixes seen at a specific depth."""
        return list(self.nodes_by_depth.get(depth, {}).keys())

    def get_nodes_at_depth(self, depth: int) -> List[TrieNode]:
        """Get all nodes at a specific depth."""
        return list(self.nodes_by_depth.get(depth, {}).values())

    def count_unique_prefixes_at_depth(self, depth: int) -> int:
        """Count unique prefixes at a specific depth."""
        return len(self.nodes_by_depth.get(depth, {}))

    def get_tofs(self, prefix: str, depth: int) -> Optional[datetime]:
        """
        Get time of first seen for a prefix at a specific depth (§4.4).

        Args:
            prefix: Prefix string
            depth: Depth to query

        Returns:
            TOFS timestamp, or None if prefix not found
        """
        node = self.get_node(prefix, depth)
        return node.tofs if node else None

    def get_unique_ip_count(self, prefix: str, depth: int) -> int:
        """
        Get unique IP count for a prefix (§4.4).

        Args:
            prefix: Prefix string
            depth: Depth to query

        Returns:
            Number of unique IPs that hit this prefix
        """
        node = self.get_node(prefix, depth)
        return node.unique_ip_count() if node else 0

    def get_error_rate(self, prefix: str, depth: int) -> float:
        """
        Get error rate for a prefix (§4.4).

        Args:
            prefix: Prefix string
            depth: Depth to query

        Returns:
            Error rate (4xx + 5xx) / total
        """
        node = self.get_node(prefix, depth)
        return node.error_rate() if node else 0.0

    def extract_prefix(self, path: str, target_depth: int) -> str:
        """
        Extract prefix from path at target depth.

        Args:
            path: Full path
            target_depth: Desired prefix depth

        Returns:
            Prefix string (truncated to target_depth)
        """
        return path[:target_depth] if len(path) >= target_depth else path


class EntityPrefixTrie:
    """
    Per-entity (IP, UA) prefix tries (§2.1).
    Tracks prefix patterns per entity to detect scanning behavior.
    """

    def __init__(self, depths: List[int]):
        """Initialize entity-specific trie collection."""
        self.depths = depths
        # Entity -> PrefixTrie mapping
        self.entity_tries: Dict[str, PrefixTrie] = {}

    def add_request(
        self,
        entity_id: str,
        path: str,
        ip: str,
        ua: Optional[str],
        status_class: str,
        timestamp: datetime,
    ):
        """
        Add a request for a specific entity.

        Args:
            entity_id: Entity identifier (IP address or UA string)
            path: Encrypted path
            ip: Client IP
            ua: User agent
            status_class: Status class
            timestamp: Request timestamp
        """
        if entity_id not in self.entity_tries:
            self.entity_tries[entity_id] = PrefixTrie(self.depths)

        self.entity_tries[entity_id].add_path(path, ip, ua, status_class, timestamp)

    def get_trie(self, entity_id: str) -> Optional[PrefixTrie]:
        """Get trie for a specific entity."""
        return self.entity_tries.get(entity_id)

    def get_all_entities(self) -> List[str]:
        """Get all entity IDs tracked."""
        return list(self.entity_tries.keys())


if __name__ == "__main__":
    # Test prefix trie
    from datetime import datetime

    trie = PrefixTrie(depths=[2, 4, 6, 8])

    # Simulate requests with common prefix
    ts = datetime.now()
    trie.add_path("/4cHCCP5v8YP_K73rMwVH0iMrEqabfCXR", "192.168.1.1", "Mozilla/5.0", "3xx", ts)
    trie.add_path("/4cHCCP5v8YP_K73rMwVH0iMrAjw9jJp7", "192.168.1.2", "curl/8.0", "2xx", ts)
    trie.add_path("/4cHCCP5v8YP_K73rMwVH0iMrEqabfCXR", "192.168.1.1", "Mozilla/5.0", "4xx", ts)

    print(f"Unique prefixes at depth 8: {trie.count_unique_prefixes_at_depth(8)}")

    # Check common prefix
    node = trie.get_node("/4cHCCP5v8YP_K73rMwVH0iMr", 24)
    if node:
        print(
            f"Prefix '/4cHCCP5v8YP_K73rMwVH0iMr' (depth 24): {node.count} hits, {node.unique_ip_count()} IPs"
        )
