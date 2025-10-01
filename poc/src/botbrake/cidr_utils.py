"""
CIDR utilities for network prefix aggregation.
"""

import ipaddress
from typing import Union


def extract_cidr_prefix(ip_str: str, ipv4_prefix: int = 24, ipv6_prefix: int = 48) -> str:
    """
    Extract CIDR prefix from IP address.

    Args:
        ip_str: IP address as string
        ipv4_prefix: Prefix length for IPv4 (default /24)
        ipv6_prefix: Prefix length for IPv6 (default /48)

    Returns:
        CIDR prefix string (e.g., "192.168.1.0/24" or "2a03:2880:f804::/48")
    """
    try:
        ip = ipaddress.ip_address(ip_str)

        if isinstance(ip, ipaddress.IPv4Address):
            # IPv4: extract /24 prefix
            network = ipaddress.ip_network(f"{ip}/{ipv4_prefix}", strict=False)
            return str(network)
        else:
            # IPv6: extract /48 prefix
            network = ipaddress.ip_network(f"{ip}/{ipv6_prefix}", strict=False)
            return str(network)
    except ValueError:
        # Invalid IP, return as-is
        return ip_str


def normalize_cidr_prefix(cidr_str: str) -> str:
    """
    Normalize CIDR prefix string.

    Args:
        cidr_str: CIDR string (e.g., "2a03:2880:f804::/48")

    Returns:
        Normalized CIDR string
    """
    try:
        network = ipaddress.ip_network(cidr_str, strict=False)
        return str(network)
    except ValueError:
        return cidr_str


def is_ipv6(ip_str: str) -> bool:
    """
    Check if IP address is IPv6.

    Args:
        ip_str: IP address string

    Returns:
        True if IPv6, False otherwise
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return isinstance(ip, ipaddress.IPv6Address)
    except ValueError:
        return False


if __name__ == "__main__":
    # Test CIDR extraction
    test_ips = [
        "192.168.1.100",
        "2a03:2880:f804:c::",
        "2a03:2880:f806:15::",
        "93.190.138.45",
    ]

    for ip in test_ips:
        prefix = extract_cidr_prefix(ip)
        print(f"{ip} -> {prefix}")