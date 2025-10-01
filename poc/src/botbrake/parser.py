"""
Apache Combined and BunnyCDN log parser with normalization (§1).
Parses HTTP logs and extracts required fields with robust error handling.
"""

import re
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from ipaddress import ip_address, IPv4Address
from enum import Enum


class LogFormat(Enum):
    """Supported log formats."""

    APACHE_COMBINED = "apache"
    BUNNYCDN = "bunnycdn"


@dataclass
class LogRecord:
    """Normalized log record structure (§1.1)."""

    ts: datetime  # Timestamp
    ip: str  # Client IP (normalized)
    ip_int: int  # IP as integer for CIDR math
    verb: str  # HTTP method (normalized to uppercase)
    path_enc: str  # URICrypt encrypted path
    status: int  # HTTP status code
    status_class: str  # Coarse class: 2xx, 3xx, 4xx, 5xx
    ref_enc: Optional[str]  # URICrypt encrypted referrer (or None)
    ua: Optional[str]  # User agent (or None)
    extras: Dict[str, Any]  # Additional fields


# Apache Combined Log Format regex
# Format: IP - - [timestamp] "METHOD path HTTP/version" status bytes "referer" "user-agent"
COMBINED_LOG_REGEX = re.compile(
    r"^(?P<ip>[^\s]+)\s+"  # IP address
    r"-\s+-\s+"  # identd, userid (usually -)
    r"\[(?P<timestamp>[^\]]+)\]\s+"  # [timestamp]
    r'"(?P<method>[A-Z]+)\s+'  # "METHOD
    r"(?P<path>[^\s]+)\s+"  # path
    r'HTTP/[^"]+"\s+'  # HTTP/version"
    r"(?P<status>\d+)\s+"  # status code
    r"(?P<bytes>\d+|-)\s+"  # bytes (or -)
    r'"(?P<referer>[^"]*)"\s+'  # "referer"
    r'"(?P<useragent>[^"]*)"'  # "user-agent"
)

# Alternative: handle lines that might not match exactly (e.g., log rotation messages)
LOG_ROTATION_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}T.*newsyslog")

# BunnyCDN Log Format regex
# Format: HIT|200|1756631440572|1314|4029805|111.187.127.0|-|https://...|LA|dnscrypt-proxy|hash|CN
# Fields: cache_status|status|timestamp_ms|bytes|zone_id|ip|referer|url|edge|user_agent|request_id|country
BUNNYCDN_LOG_REGEX = re.compile(
    r"^(?P<cache_status>[^|]+)\|"  # Cache status (HIT/MISS)
    r"(?P<status>\d+)\|"  # HTTP status code
    r"(?P<timestamp>\d+)\|"  # Timestamp in milliseconds
    r"(?P<bytes>\d+)\|"  # Response size in bytes
    r"(?P<zone_id>\d+)\|"  # Pull zone ID
    r"(?P<ip>[^|]+)\|"  # Client IP
    r"(?P<referer>[^|]*)\|"  # Referer (or -)
    r"(?P<url>[^|]+)\|"  # Full URL
    r"(?P<edge>[^|]+)\|"  # Edge location
    r"(?P<useragent>[^|]+)\|"  # User agent
    r"(?P<request_id>[^|]+)\|"  # Request ID/hash
    r"(?P<country>[^|]+)$"  # Country code
)


def normalize_ip(ip_str: str) -> tuple[str, int]:
    """
    Normalize IP address (§1).
    - IPv4: as-is
    - IPv6: compressed form
    Returns: (normalized_str, integer_representation)
    """
    try:
        ip_obj = ip_address(ip_str)
        if isinstance(ip_obj, IPv4Address):
            return str(ip_obj), int(ip_obj)
        else:  # IPv6
            return ip_obj.compressed, int(ip_obj)
    except ValueError:
        # Invalid IP, return as-is with 0
        return ip_str, 0


def normalize_status(status: int) -> str:
    """Normalize status code into coarse class (§1)."""
    if 200 <= status < 300:
        return "2xx"
    elif 300 <= status < 400:
        return "3xx"
    elif 400 <= status < 500:
        return "4xx"
    elif 500 <= status < 600:
        return "5xx"
    else:
        return "other"


def normalize_verb(verb: str) -> str:
    """Normalize HTTP method to uppercase (§1)."""
    verb = verb.upper()
    # Map uncommon methods to OTHER if needed (for now, keep all)
    return verb


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parse Apache log timestamp.
    Format: 25/Sep/2025:19:00:01 +0000
    """
    try:
        # Remove timezone for simplicity (assume UTC)
        ts_str = ts_str.split()[0]  # "25/Sep/2025:19:00:01"
        return datetime.strptime(ts_str, "%d/%b/%Y:%H:%M:%S")
    except (ValueError, IndexError):
        return None


def parse_bunnycdn_timestamp(ts_ms: str) -> Optional[datetime]:
    """
    Parse BunnyCDN timestamp in milliseconds.
    Format: 1756631440572 (Unix timestamp in milliseconds)
    """
    try:
        ts_sec = int(ts_ms) / 1000.0
        return datetime.fromtimestamp(ts_sec)
    except (ValueError, OSError):
        return None


def parse_bunnycdn_line(line: str) -> Optional[LogRecord]:
    """
    Parse a single BunnyCDN log line.
    Returns LogRecord or None if unparseable.
    """
    line = line.strip()
    if not line:
        return None

    match = BUNNYCDN_LOG_REGEX.match(line)
    if not match:
        return None

    try:
        # Extract fields
        ip_str = match.group("ip")
        timestamp_ms = match.group("timestamp")
        status_str = match.group("status")
        url = match.group("url")
        referer = match.group("referer")
        useragent = match.group("useragent")

        # Normalize
        ip_norm, ip_int = normalize_ip(ip_str)
        ts = parse_bunnycdn_timestamp(timestamp_ms)
        if ts is None:
            return None

        status = int(status_str)
        status_class = normalize_status(status)

        # BunnyCDN logs don't have HTTP verb in the log, assume GET
        verb = "GET"

        # Handle empty/missing referer and UA
        ref_enc = None if referer in ("", "-") else referer
        ua = None if useragent in ("", "-") else useragent

        # Extract path from URL
        # URL format: https://download.dnscrypt.info/resolvers-list/v3/public-resolvers.md.minisig
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            path = parsed.path if parsed.path else "/"
        except Exception:
            path = url

        return LogRecord(
            ts=ts,
            ip=ip_norm,
            ip_int=ip_int,
            verb=verb,
            path_enc=path,
            status=status,
            status_class=status_class,
            ref_enc=ref_enc,
            ua=ua,
            extras={
                "cache_status": match.group("cache_status"),
                "edge": match.group("edge"),
                "country": match.group("country"),
                "request_id": match.group("request_id"),
            },
        )
    except (ValueError, AttributeError):
        return None


def detect_log_format(line: str) -> Optional[LogFormat]:
    """
    Auto-detect log format from a sample line.
    Returns LogFormat or None if unrecognizable.
    """
    line = line.strip()
    if not line:
        return None

    # Check for BunnyCDN format (pipe-delimited with specific structure)
    if BUNNYCDN_LOG_REGEX.match(line):
        return LogFormat.BUNNYCDN

    # Check for Apache Combined format
    if COMBINED_LOG_REGEX.match(line):
        return LogFormat.APACHE_COMBINED

    return None


def parse_log_line(line: str, log_format: Optional[LogFormat] = None) -> Optional[LogRecord]:
    """
    Parse a single log line (§1.1).
    Auto-detects format if not specified.
    Returns LogRecord or None if unparseable.
    """
    # Auto-detect format if not specified
    if log_format is None:
        log_format = detect_log_format(line)
        if log_format is None:
            return None

    if log_format == LogFormat.BUNNYCDN:
        return parse_bunnycdn_line(line)

    # Apache Combined format parsing
    line = line.strip()
    if not line:
        return None

    # Check for log rotation message
    if LOG_ROTATION_REGEX.match(line):
        return None

    # Match against combined log format
    match = COMBINED_LOG_REGEX.match(line)
    if not match:
        return None

    try:
        # Extract fields
        ip_str = match.group("ip")
        timestamp_str = match.group("timestamp")
        method = match.group("method")
        path = match.group("path")
        status_str = match.group("status")
        referer = match.group("referer")
        useragent = match.group("useragent")

        # Normalize
        ip_norm, ip_int = normalize_ip(ip_str)
        ts = parse_timestamp(timestamp_str)
        if ts is None:
            return None

        status = int(status_str)
        status_class = normalize_status(status)
        verb = normalize_verb(method)

        # Handle empty/missing referer and UA
        ref_enc = None if referer in ("", "-") else referer
        ua = None if useragent in ("", "-") else useragent

        return LogRecord(
            ts=ts,
            ip=ip_norm,
            ip_int=ip_int,
            verb=verb,
            path_enc=path,
            status=status,
            status_class=status_class,
            ref_enc=ref_enc,
            ua=ua,
            extras={},
        )
    except (ValueError, AttributeError):
        return None


def is_error_status(status_class: str) -> bool:
    """Check if status is an error (4xx or 5xx)."""
    return status_class in ("4xx", "5xx")


if __name__ == "__main__":
    # Test parser with sample lines
    apache_test_lines = [
        '46.42.143.239 - - [25/Sep/2025:19:00:01 +0000] "GET /4cHCCP5v8YP_K73rMwVH0iMr HTTP/1.1" 301 169 "-" "OpenWebReader/0.3a1"',
        '154.249.137.8 - - [25/Sep/2025:19:04:21 +0000] "GET /4cHCCP5v8YP_K73rMwVH0iMrEqabfCXRkjOgIq97EtagZjzrtplqVDF37CJ25wyR9VXKr HTTP/1.0" 408 0 "-" "-"',
    ]

    bunnycdn_test_lines = [
        "HIT|200|1756631440572|1314|4029805|111.187.127.0|-|https://download.dnscrypt.info/resolvers-list/v3/public-resolvers.md.minisig|LA|dnscrypt-proxy|b8cae46995f5c009698ced1a24e1ba02|CN",
        "HIT|200|1756631439660|148180|4029805|111.187.127.0|-|https://download.dnscrypt.info/resolvers-list/v3/public-resolvers.md|LA|dnscrypt-proxy|af13b4e66568d6e004ee44bf86c9f205|CN",
    ]

    print("Testing Apache Combined format:")
    for line in apache_test_lines:
        record = parse_log_line(line, LogFormat.APACHE_COMBINED)
        if record:
            print(
                f"IP: {record.ip}, Status: {record.status} ({record.status_class}), UA: {record.ua}"
            )
            print(f"Path prefix (8): {record.path_enc[:8]}")
        else:
            print("Failed to parse")

    print("\nTesting BunnyCDN format:")
    for line in bunnycdn_test_lines:
        record = parse_log_line(line, LogFormat.BUNNYCDN)
        if record:
            print(
                f"IP: {record.ip}, Status: {record.status} ({record.status_class}), UA: {record.ua}"
            )
            print(f"Path: {record.path_enc}, Country: {record.extras.get('country')}")
        else:
            print("Failed to parse")
