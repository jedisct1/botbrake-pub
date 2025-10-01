#!/usr/bin/env python3
"""
BotBrake: URICrypt-aware HTTP log abuse detection system.
Main entry point implementing two-pass training + detection (§9).

Usage: python3 main.py <access_log_file>
"""

import sys
import json
from datetime import timedelta
from collections import defaultdict
from typing import List, Optional

from botbrake.config import DEFAULT_CONFIG, DetectionConfig
from botbrake.parser import parse_log_line, LogRecord, LogFormat, detect_log_format
from botbrake.prefix_trie import PrefixTrie, EntityPrefixTrie
from botbrake.entity_tracker import EntityTracker, PersistenceTracker
from botbrake.statistics import BaselineStatistics, BetaBinomialPrior
from botbrake.detector import Detector, DetectionDecision
from botbrake.rate_tracker import RateTracker


class BotBrake:
    """
    Main detection system implementing two-pass algorithm.
    """

    def __init__(self, config: DetectionConfig, log_format: Optional[LogFormat] = None):
        """Initialize BotBrake with configuration."""
        self.config = config
        self.log_format = log_format  # None means auto-detect
        self.all_records: List[LogRecord] = []

    def load_logs(self, log_file: str) -> int:
        """
        Load and parse all log records.

        Args:
            log_file: Path to log file

        Returns:
            Number of successfully parsed records
        """
        print(f"Loading logs from {log_file}...")
        count = 0

        # Auto-detect format from first few lines if not specified
        detected_format = self.log_format
        if detected_format is None:
            print("Auto-detecting log format...")
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Check first 10 lines
                        break
                    fmt = detect_log_format(line)
                    if fmt:
                        detected_format = fmt
                        print(f"  Detected format: {fmt.value}")
                        break

            if detected_format is None:
                print("  Warning: Could not auto-detect format, using Apache Combined")
                detected_format = LogFormat.APACHE_COMBINED

        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                if line_no % 10000 == 0:
                    print(f"  Parsed {line_no} lines...", end="\r")

                record = parse_log_line(line, detected_format)
                if record:
                    self.all_records.append(record)
                    count += 1

        print(f"\nLoaded {count} valid log records")

        # Sort records by timestamp (logs may be in reverse order)
        if self.all_records:
            self.all_records.sort(key=lambda r: r.ts)
            print(
                f"Time range: {self.all_records[0].ts} to {self.all_records[-1].ts}"
            )

        return count

    def training_pass(self) -> BaselineStatistics:
        """
        Pass 1: Learn baselines from all data (§8.1).

        Returns:
            BaselineStatistics with learned parameters
        """
        print("\n=== Training Pass: Learning Baselines ===")

        baseline = BaselineStatistics()

        # Build global structures
        global_trie = PrefixTrie(self.config.prefix.depths)
        entity_tracker = EntityTracker()

        # Simple time-based windowing: use first hour of data as training window
        if not self.all_records:
            return baseline

        start_time = self.all_records[0].ts
        end_time = start_time + timedelta(hours=1)

        # Process records in training window
        entity_error_rates = []
        entity_explore_ratios = []
        prefix_fanouts = defaultdict(list)
        entity_depth_scores = []  # §2.3: Track depth scores for baseline

        print("Building training data structures...")
        for record in self.all_records:
            if record.ts > end_time:
                break  # Only use first hour for training

            # Update global trie
            global_trie.add_path(
                record.path_enc, record.ip, record.ua, record.status_class, record.ts
            )

            # Update entity metrics (IP)
            metrics = entity_tracker.get_or_create_metrics(
                "ip", record.ip, "training", start_time, end_time
            )
            metrics.add_request(
                record.path_enc,
                record.status_class,
                record.verb,
                record.ts,
                ref=record.ref_enc,
                ua=record.ua,
            )

            # Track prefixes and depth (§2.3)
            deepest_depth = 0
            for depth in self.config.prefix.depths:
                if len(record.path_enc) >= depth:
                    metrics.add_prefix(depth, record.path_enc[:depth])
                    deepest_depth = depth  # Track deepest depth reached
            if deepest_depth > 0:
                metrics.add_depth(deepest_depth)

        # Extract baseline metrics from entities with sufficient volume
        print("Extracting baseline statistics...")
        n_min_training = 30  # Lower threshold for training data

        for ip in entity_tracker.get_all_entities("ip"):
            metrics = entity_tracker.get_metrics("ip", ip, "training")
            if metrics and metrics.total_requests >= n_min_training:
                # Only use entities with low-ish error rates as "legitimate"
                error_rate = metrics.error_rate()
                if error_rate < 0.5:  # Filter out obvious abusers
                    entity_error_rates.append(error_rate)

                # Exploration metrics
                explore_ratio = metrics.explore_ratio()
                if explore_ratio > 0:
                    entity_explore_ratios.append(explore_ratio)

                # Prefix fanouts
                for depth in self.config.prefix.depths:
                    fanout = metrics.prefix_fanout(depth)
                    if fanout > 0:
                        prefix_fanouts[depth].append(fanout)

                # Depth scores (§2.3)
                depth_score = metrics.compute_depth_score()
                if depth_score > 0:
                    entity_depth_scores.append(depth_score)

        # Compute global statistics
        if entity_error_rates:
            import numpy as np

            baseline.global_error_rate = float(np.median(entity_error_rates))
            print(f"  Global error rate (median): {baseline.global_error_rate:.3f}")

            # Fit Beta-Binomial prior
            baseline.beta_prior = BetaBinomialPrior.from_error_rates(entity_error_rates)
            print(
                f"  Beta prior: α={baseline.beta_prior.alpha:.2f}, β={baseline.beta_prior.beta:.2f}"
            )
        else:
            baseline.global_error_rate = 0.08  # Fallback
            print("  Warning: Insufficient data, using default error rate 0.08")

        # Store exploration baselines
        for val in entity_explore_ratios:
            baseline.add_explore_ratio(val)
        print(f"  Exploration ratios: {len(entity_explore_ratios)} samples")

        for depth, values in prefix_fanouts.items():
            for val in values:
                baseline.add_prefix_fanout(depth, val)
        print(f"  Prefix fanout baselines: {len(prefix_fanouts)} depths")

        # Store depth score baselines (§2.3)
        for val in entity_depth_scores:
            baseline.add_depth_score(val)
        print(f"  Depth scores: {len(entity_depth_scores)} samples")

        return baseline

    def detection_pass(self, baseline: BaselineStatistics) -> List[DetectionDecision]:
        """
        Pass 2: Detect abuse using learned baselines (§9).

        Args:
            baseline: Learned baseline statistics

        Returns:
            List of detection decisions
        """
        print("\n=== Detection Pass: Analyzing Traffic ===")

        # Initialize structures
        global_trie = PrefixTrie(self.config.prefix.depths)
        entity_tries = {
            "ip": EntityPrefixTrie(self.config.prefix.depths),
            "ua": EntityPrefixTrie(self.config.prefix.depths),
        }
        entity_tracker = EntityTracker()
        persistence_tracker = PersistenceTracker()

        # Create rate tracker for burst detection (§4.3)
        rate_tracker = RateTracker(
            alpha=0.3,  # EWMA smoothing
            cusum_k=2.0,  # CUSUM slack
            cusum_h=5.0,  # CUSUM threshold
        )
        rate_tracker.set_training_mode(False)  # Enable detection mode

        detector = Detector(self.config, baseline, rate_tracker)

        decisions = []

        # Create time windows
        if not self.all_records:
            return decisions

        start_time = self.all_records[0].ts
        end_time = self.all_records[-1].ts

        # Use 5-minute windows for detection
        window_duration = timedelta(seconds=300)  # 5 minutes
        current_window_start = start_time
        window_id = 0

        print(f"Processing {len(self.all_records)} records in 5-minute windows...")

        while current_window_start < end_time:
            current_window_end = current_window_start + window_duration
            window_name = f"5m_{window_id}"

            # Process records in this window
            records_in_window = [
                r for r in self.all_records if current_window_start <= r.ts < current_window_end
            ]

            if records_in_window:
                # Update structures
                for record in records_in_window:
                    # Global trie
                    global_trie.add_path(
                        record.path_enc, record.ip, record.ua, record.status_class, record.ts
                    )

                    # Entity tries
                    if record.ip:
                        entity_tries["ip"].add_request(
                            record.ip,
                            record.path_enc,
                            record.ip,
                            record.ua,
                            record.status_class,
                            record.ts,
                        )
                    if record.ua:
                        entity_tries["ua"].add_request(
                            record.ua,
                            record.path_enc,
                            record.ip,
                            record.ua,
                            record.status_class,
                            record.ts,
                        )

                    # Entity metrics for IPs
                    ip_metrics = entity_tracker.get_or_create_metrics(
                        "ip", record.ip, window_name, current_window_start, current_window_end
                    )
                    ip_metrics.add_request(
                        record.path_enc,
                        record.status_class,
                        record.verb,
                        record.ts,
                        ref=record.ref_enc,
                        ua=record.ua,
                        ip=record.ip,
                    )

                    # Track prefixes and depth (§2.3)
                    deepest_depth = 0
                    for depth in self.config.prefix.depths:
                        if len(record.path_enc) >= depth:
                            ip_metrics.add_prefix(depth, record.path_enc[:depth])
                            deepest_depth = depth
                    if deepest_depth > 0:
                        ip_metrics.add_depth(deepest_depth)

                    # Entity metrics for UAs
                    if record.ua:
                        ua_metrics = entity_tracker.get_or_create_metrics(
                            "ua", record.ua, window_name, current_window_start, current_window_end
                        )
                        ua_metrics.add_request(
                            record.path_enc,
                            record.status_class,
                            record.verb,
                            record.ts,
                            ref=record.ref_enc,
                            ua=record.ua,
                            ip=record.ip,
                        )

                        # Track prefixes and depth for UA (§2.3)
                        deepest_depth_ua = 0
                        for depth in self.config.prefix.depths:
                            if len(record.path_enc) >= depth:
                                ua_metrics.add_prefix(depth, record.path_enc[:depth])
                                deepest_depth_ua = depth
                        if deepest_depth_ua > 0:
                            ua_metrics.add_depth(deepest_depth_ua)

                # Calculate total window requests for dominance detection
                total_window_requests = len(records_in_window)

                # Update rate tracker with entity rates (§4.3)
                window_duration_sec = window_duration.total_seconds()
                for entity_type in ["ip", "ua"]:
                    for entity_value in entity_tracker.get_all_entities(entity_type):
                        metrics = entity_tracker.get_metrics(entity_type, entity_value, window_name)
                        if metrics:
                            rate_tracker.update_rate(
                                entity_type,
                                entity_value,
                                window_duration_sec,
                                metrics.total_requests,
                                current_window_end,
                            )

                # Analyze CIDR networks first (aggregated from IPs)
                cidr_metrics_map = entity_tracker.aggregate_cidr_metrics(
                    window_name,
                    current_window_start,
                    current_window_end,
                    ipv4_prefix=self.config.cidr.ipv4_test_prefix,
                    ipv6_prefix=self.config.cidr.ipv6_test_prefix,
                )

                for cidr_prefix, cidr_metrics in cidr_metrics_map.items():
                    # Get persistence count
                    persist_count = persistence_tracker.get_consecutive_windows("cidr", cidr_prefix)

                    # Analyze CIDR with dominance detection
                    decision = detector.analyze_entity(
                        cidr_metrics,
                        global_trie,
                        persist_count,
                        window_name,
                        total_window_requests,
                        window_duration_sec,
                    )

                    # Record persistence
                    has_signal = decision is not None
                    persistence_tracker.record_signal("cidr", cidr_prefix, window_name, has_signal)

                    if decision:
                        decisions.append(decision)

                # Analyze individual entities (IPs and UAs)
                for entity_type in ["ip", "ua"]:
                    for entity_value in entity_tracker.get_all_entities(entity_type):
                        metrics = entity_tracker.get_metrics(entity_type, entity_value, window_name)
                        if not metrics:
                            continue

                        # Get persistence count
                        persist_count = persistence_tracker.get_consecutive_windows(
                            entity_type, entity_value
                        )

                        # Analyze (with total window requests for potential dominance)
                        decision = detector.analyze_entity(
                            metrics,
                            global_trie,
                            persist_count,
                            window_name,
                            total_window_requests,
                            window_duration_sec,
                        )

                        # Record persistence
                        has_signal = decision is not None
                        persistence_tracker.record_signal(
                            entity_type, entity_value, window_name, has_signal
                        )

                        if decision:
                            decisions.append(decision)

            # Prune old entities from rate tracker (§4.3 memory management)
            if window_id % 12 == 0:  # Every hour (12 × 5min windows)
                rate_tracker.prune_old_entities(current_window_end)

            window_id += 1
            current_window_start = current_window_end

            if window_id % 10 == 0:
                print(
                    f"  Processed {window_id} windows, {len(decisions)} decisions so far...",
                    end="\r",
                )

        print(f"\nDetection complete: {len(decisions)} block decisions")
        return decisions

    def run(self, log_file: str, output_file: str = "tmp/decisions.json"):
        """
        Run full two-pass detection.

        Args:
            log_file: Input log file
            output_file: Output JSON file for decisions
        """
        # Load logs
        if not self.load_logs(log_file):
            print("Error: No valid log records found")
            return

        # Training pass
        baseline = self.training_pass()

        # Detection pass
        decisions = self.detection_pass(baseline)

        # Output results
        self.write_output(decisions, output_file)
        self.print_summary(decisions)

    def write_output(self, decisions: List[DetectionDecision], output_file: str):
        """Write decisions to JSON file (§13)."""
        print(f"\nWriting decisions to {output_file}...")

        output_data = []
        for decision in decisions:
            output_data.append(
                {
                    "when": decision.when,
                    "entity_type": decision.entity_type,
                    "entity_value": decision.entity_value,
                    "score": round(decision.score, 2),
                    "recommended_action": decision.recommended_action,
                    "duration_sec": decision.duration_sec,
                    "signals": decision.signals,
                    "dampeners": decision.dampeners,
                    "explanation": decision.explanation,
                }
            )

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Wrote {len(output_data)} decisions")

    def print_summary(self, decisions: List[DetectionDecision]):
        """Print detection summary."""
        print("\n=== Detection Summary ===")

        if not decisions:
            print("No entities flagged for blocking.")
            return

        # Group by entity type, then aggregate by entity value
        by_type = defaultdict(lambda: defaultdict(list))
        for d in decisions:
            by_type[d.entity_type][d.entity_value].append(d)

        for entity_type, entities in by_type.items():
            print(f"\n{entity_type.upper()}:")

            # Aggregate stats for each unique entity
            entity_stats = []
            for entity_value, decs in entities.items():
                max_score = max(d.score for d in decs)
                avg_score = sum(d.score for d in decs) / len(decs)
                num_windows = len(decs)
                max_duration = max(d.duration_sec for d in decs)
                # Get explanation from highest-scoring detection
                best_dec = max(decs, key=lambda d: d.score)

                entity_stats.append({
                    'value': entity_value,
                    'max_score': max_score,
                    'avg_score': avg_score,
                    'num_windows': num_windows,
                    'max_duration': max_duration,
                    'explanation': best_dec.explanation,
                })

            # Sort by max score descending
            entity_stats.sort(key=lambda x: x['max_score'], reverse=True)

            # Print unique entities (top 10)
            for stat in entity_stats[:10]:
                print(f"  {stat['value']}:")
                print(f"    Max score: {stat['max_score']:.1f}, Avg score: {stat['avg_score']:.1f}")
                print(f"    Detected in {stat['num_windows']} window(s), recommended duration: {stat['max_duration']//60}min")
                print(f"    {stat['explanation']}")
                print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <access_log_file> [--format bunnycdn|apache]")
        sys.exit(1)

    log_file = sys.argv[1]

    # Parse log format option (None means auto-detect)
    log_format = None
    if len(sys.argv) > 2:
        if sys.argv[2] == "--format" and len(sys.argv) > 3:
            format_str = sys.argv[3].lower()
            if format_str == "bunnycdn":
                log_format = LogFormat.BUNNYCDN
            elif format_str == "apache":
                log_format = LogFormat.APACHE_COMBINED
            else:
                print(f"Unknown format: {format_str}. Will auto-detect.")
                log_format = None

    # Run detection
    botbrake = BotBrake(DEFAULT_CONFIG, log_format)
    botbrake.run(log_file)


if __name__ == "__main__":
    main()
