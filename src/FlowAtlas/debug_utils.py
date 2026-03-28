"""
Debug utilities for Wave Function Collapse and related procedural generation algorithms.

This module separates debug instrumentation from core algorithm logic, providing:
- DebugStatisticsCollector: data collection during generation
- DebugReportFormatter: output formatting for debug reports
- Standalone distribution metric functions for reuse

The module can be used standalone or integrated into generators via settings dicts.
All debug features are optional and can be toggled via settings flags.

Future note:
- If debugging needs expand significantly, investigate richer tooling such as
    structured logging or terminal formatting libraries instead of continuing to
    grow this custom formatter layer indefinitely.
"""

from typing import Optional


# ===== DISTRIBUTION METRICS (Standalone utilities) =====

def normalize_distribution(dist: dict) -> dict:
    """
    Normalize a distribution dict to probabilities (sum to 1.0).

    Args:
        dist: {terrain_name: weight, ...} - raw frequency weights

    Returns:
        {terrain_name: probability, ...} - normalized to sum to 1.0

    Notes:
        - Empty distributions return empty dict
        - Zero-weight distributions return uniform distribution
    """
    if not dist:
        return {}
    total = sum(dist.values())
    if total <= 0:
        uniform = 1.0 / len(dist)
        return {key: uniform for key in dist}
    return {key: value / total for key, value in dist.items()}


def distribution_gap_metrics(observed: dict, target: dict, domain: tuple) -> dict:
    """
    Compute gap metrics between observed and target distributions.

    Metrics include:
    - Per-terrain breakdown (observed, target, delta, match scores)
    - Aggregate gap measures (TV distance, MAE, RMSE)
    - Match quality scores (equal-weight and target-weighted)

    Args:
        observed: {terrain: count_or_weight, ...} - actual distribution
        target: {terrain: count_or_weight, ...} - desired distribution
        domain: tuple of all terrain names in the domain (for normalization)

    Returns:
        dict with:
            'observed': normalized observed probabilities
            'target': normalized target probabilities
            'per_terrain': {terrain: {'observed': p, 'target': p, 'delta': d, ...}}
            'tv_distance': total variation distance (0.5 * L1)
            'equal_weight_match': match score (1.0 = perfect, 0.0 = worst)
            'target_weighted_match': match score weighted by target distribution
            (plus additional metrics: max_abs_error, mean_abs_error, l1_distance, rmse)
    """
    observed_prob = normalize_distribution(observed)
    target_prob = normalize_distribution(target)

    l1_distance = 0.0
    max_abs_error = 0.0
    squared_sum = 0.0
    weighted_abs_error = 0.0
    per_terrain = {}

    for terrain in domain:
        obs = observed_prob.get(terrain, 0.0)
        exp = target_prob.get(terrain, 0.0)
        signed_diff = obs - exp
        abs_diff = abs(signed_diff)

        l1_distance += abs_diff
        squared_sum += abs_diff * abs_diff
        weighted_abs_error += abs_diff * exp
        if abs_diff > max_abs_error:
            max_abs_error = abs_diff

        per_terrain[terrain] = {
            'observed': obs,
            'target': exp,
            'delta': signed_diff,
            'abs_delta': abs_diff,
            'match': max(0.0, 1.0 - abs_diff),
            'target_weight': exp,
        }

    rmse = (squared_sum / len(domain)) ** 0.5 if domain else 0.0
    mean_abs_error = (l1_distance / len(domain)) if domain else 0.0
    equal_weight_match = max(0.0, 1.0 - mean_abs_error)
    target_weighted_match = max(0.0, 1.0 - weighted_abs_error)

    return {
        'observed': observed_prob,
        'target': target_prob,
        'per_terrain': per_terrain,
        'l1_distance': l1_distance,
        'tv_distance': 0.5 * l1_distance,
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'equal_weight_match': equal_weight_match,
        'target_weighted_match': target_weighted_match,
        'rmse': rmse,
    }


# ===== DEBUG STATISTICS COLLECTOR =====

class DebugStatisticsCollector:
    """
    Collects debug statistics during generation without coupling to algorithm logic.

    This class manages:
    - Per-step data collection (terrains assigned, timing, flags/weights)
    - Windowed aggregation (every N steps for progress reports)
    - Final distribution analysis (comparing observed vs target distributions)
    - Rule-specific metrics (per DistRule, split by province/border)

    Usage pattern:
        1. Initialize with generator's graph and rule_managers
        2. Call record_step() after each assignment
        3. Call flush_window() periodically or after complete
        4. Access via get_statistics() for data, or use formatter for output
    """

    def __init__(self, graph, rule_managers: list, store_step_snapshots: bool = True):
        """
        Initialize collector for a specific generation run.

        Args:
            graph: TerrainGraph with global_metrics['terrain_domain'] set
            rule_managers: list of RuleManager objects (may contain DistRule instances)
            store_step_snapshots: whether to retain per-step snapshot history
        """
        self.graph = graph
        self.rule_managers = rule_managers
        self.store_step_snapshots = store_step_snapshots
        self.stats = self._make_empty_stats()

    def _make_empty_stats(self) -> dict:
        """Create initial empty statistics dictionary."""
        return {
            'enabled': True,
            'steps': 0,
            'timing': {
                'total_seconds': 0.0,
                'province_assignment_seconds': 0.0,
                'border_assignment_seconds': 0.0,
                'mean_step_seconds': 0.0,
            },
            'progress': [],
            'progress_windows': [],
            'window_state': {
                'start_step': 1,
                'steps': 0,
                'timing_sum_seconds': 0.0,
                'assignment_counts': {
                    'province': {},
                    'border': {},
                },
                'rule_windows': {},
            },
            'final_report': {},
        }

    def _iter_dist_rules(self):
        """Iterate over all DistRule instances in rule managers."""
        for manager in self.rule_managers:
            for rule in manager.rules:
                # Check if rule is a DistRule (has target distributions)
                if hasattr(rule, 'target_province_dist') and hasattr(rule, 'target_border_dist'):
                    yield manager, rule

    def _ensure_rule_window(self, rule_key: str, manager_name: str) -> dict:
        """Ensure rule window entry exists in current window_state."""
        rule_windows = self.stats['window_state']['rule_windows']
        if rule_key in rule_windows:
            return rule_windows[rule_key]

        entry = {
            'manager': manager_name,
            'province': {
                'importance_sum': 0.0,
                'observed_weighted_counts': {},
                'expected_weighted_counts': {},
            },
            'border': {
                'importance_sum': 0.0,
                'observed_weighted_counts': {},
                'expected_weighted_counts': {},
            },
        }
        rule_windows[rule_key] = entry
        return entry

    def _get_progress_state(self) -> dict:
        """Capture current progress (completion %, set counts)."""
        total_provinces = self.graph.global_metrics.get('provinces', 0)
        total_borders = self.graph.global_metrics.get('borders', 0)
        set_provinces = self.graph.global_metrics.get('set_provinces', 0)
        set_borders = self.graph.global_metrics.get('set_borders', 0)
        total_elements = total_provinces + total_borders
        set_elements = set_provinces + set_borders

        return {
            'set_provinces': set_provinces,
            'set_borders': set_borders,
            'total_provinces': total_provinces,
            'total_borders': total_borders,
            'completion_ratio': (set_elements / total_elements) if total_elements else 1.0,
        }

    def get_progress_state(self) -> dict:
        """Public accessor for current progress state."""
        return self._get_progress_state()

    def record_step(self, step_info: dict, step_seconds: Optional[float] = None):
        """
        Record a single step (terrain assignment).

        Args:
            step_info: dict with 'element_kind', 'selected_terrain',
                      'selected_probability', 'element_flags'
            step_seconds: execution time for this step (or None to skip timing)
        """
        self.stats['steps'] += 1

        if step_seconds is not None:
            self.stats['timing']['total_seconds'] += step_seconds
            if step_info['element_kind'] == 'province':
                self.stats['timing']['province_assignment_seconds'] += step_seconds
            else:
                self.stats['timing']['border_assignment_seconds'] += step_seconds
            self.stats['window_state']['timing_sum_seconds'] += step_seconds

        # Count this assignment in window
        kind = step_info['element_kind']
        terrain = step_info['selected_terrain']
        window_counts = self.stats['window_state']['assignment_counts'][kind]
        window_counts[terrain] = window_counts.get(terrain, 0) + 1
        self.stats['window_state']['steps'] += 1

        # Track per-rule importance and weighted counts
        element_flags = step_info.get('element_flags', {})
        for manager, rule in self._iter_dist_rules():
            flag_weight = float(element_flags.get(rule.flag, 0.0))
            if flag_weight <= 0:
                continue

            kind_key = 'province' if kind == 'province' else 'border'
            target_raw = rule.target_province_dist if kind_key == 'province' else rule.target_border_dist
            target_prob = normalize_distribution(target_raw)

            rule_window = self._ensure_rule_window(rule.rule_key, manager.name)
            kind_window = rule_window[kind_key]
            kind_window['importance_sum'] += flag_weight
            kind_window['observed_weighted_counts'][terrain] = (
                kind_window['observed_weighted_counts'].get(terrain, 0.0) + flag_weight
            )
            for t_name, t_prob in target_prob.items():
                kind_window['expected_weighted_counts'][t_name] = (
                    kind_window['expected_weighted_counts'].get(t_name, 0.0) +
                    (flag_weight * t_prob)
                )

        if self.store_step_snapshots:
            # Store step snapshot
            progress_entry = {
                'step': self.stats['steps'],
                'element_kind': step_info['element_kind'],
                'selected_terrain': step_info['selected_terrain'],
                'selected_probability': step_info['selected_probability'],
                'step_seconds': step_seconds if step_seconds is not None else 0.0,
                **self._get_progress_state(),
            }
            self.stats['progress'].append(progress_entry)

    def _build_window_kind_report(self, kind: str, kind_domain: tuple) -> dict:
        """Build aggregated report for a kind (province/border) in current window."""
        window_counts = self.stats['window_state']['assignment_counts'][kind]
        observed_total = sum(window_counts.values())
        observed_prob = normalize_distribution({terrain: window_counts.get(terrain, 0.0) for terrain in kind_domain})

        rules_report = []
        for manager, rule in self._iter_dist_rules():
            rule_window = self._ensure_rule_window(rule.rule_key, manager.name)
            bucket = rule_window[kind]

            importance_sum = bucket['importance_sum']
            if importance_sum <= 0:
                continue

            rule_metrics = distribution_gap_metrics(
                observed=bucket['observed_weighted_counts'],
                target=bucket['expected_weighted_counts'],
                domain=kind_domain,
            )
            rules_report.append({
                'manager': manager.name,
                'rule_key': rule.rule_key,
                'importance_sum': importance_sum,
                'observed_prob': rule_metrics['observed'],
                'target_prob': rule_metrics['target'],
                'metrics': rule_metrics,
            })

        return {
            'assignments': observed_total,
            'observed_prob': observed_prob,
            'rules': rules_report,
        }

    def flush_window(self, force: bool = False) -> Optional[dict]:
        """
        Flush current window (aggregate and store if non-empty).

        Args:
            force: if True, flush even if window is empty

        Returns:
            Flushed summary dict or None
        """
        window = self.stats['window_state']
        if window['steps'] == 0 and not force:
            return None

        terrain_domain = self.graph.global_metrics['terrain_domain']
        summary = {
            'start_step': window['start_step'],
            'end_step': self.stats['steps'],
            'steps': window['steps'],
            'timing_sum_seconds': window['timing_sum_seconds'],
            **self._get_progress_state(),
            'province': self._build_window_kind_report('province', terrain_domain['province_terrains']),
            'border': self._build_window_kind_report('border', terrain_domain['border_terrains']),
        }
        self.stats['progress_windows'].append(summary)

        # Reset window state
        self.stats['window_state'] = {
            'start_step': self.stats['steps'] + 1,
            'steps': 0,
            'timing_sum_seconds': 0.0,
            'assignment_counts': {
                'province': {},
                'border': {},
            },
            'rule_windows': {},
        }

        return summary

    def build_final_report(self) -> dict:
        """
        Build final distribution analysis comparing observed vs target.

        Returns:
            Report dict with observed distributions and per-rule metrics
        """
        terrain_domain = self.graph.global_metrics['terrain_domain']

        # Collect terrain counts from currently set elements
        province_counts = {}
        border_counts = {}

        for terrain in terrain_domain['province_terrains']:
            province_counts[terrain] = 0
        for terrain in terrain_domain['border_terrains']:
            border_counts[terrain] = 0

        for element in self.graph.get_all_elements():
            terrain = element.get('terrain', None)
            if terrain is None:
                continue

            if element.is_province and terrain in province_counts:
                province_counts[terrain] += 1
            elif terrain in border_counts:
                border_counts[terrain] += 1

        report = {
            'observed': {
                'province': normalize_distribution(province_counts),
                'border': normalize_distribution(border_counts),
            },
            'rule_targets': [],
        }

        # Per-rule weighted metrics
        for manager in self.rule_managers:
            for rule in manager.rules:
                if not (hasattr(rule, 'target_province_dist') and hasattr(rule, 'target_border_dist')):
                    continue

                province_observed_weighted = {}
                province_target_weighted = {}
                border_observed_weighted = {}
                border_target_weighted = {}

                province_target_prob = normalize_distribution(rule.target_province_dist)
                border_target_prob = normalize_distribution(rule.target_border_dist)

                for element in self.graph.get_all_elements():
                    terrain = element.get('terrain', None)
                    if terrain is None:
                        continue

                    flags = element.get('flags', {})
                    flag_weight = float(flags.get(rule.flag, 0.0)) if isinstance(flags, dict) else 0.0
                    if flag_weight <= 0:
                        continue

                    if element.is_province:
                        province_observed_weighted[terrain] = (
                            province_observed_weighted.get(terrain, 0.0) + flag_weight
                        )
                        for t_name, t_prob in province_target_prob.items():
                            province_target_weighted[t_name] = (
                                province_target_weighted.get(t_name, 0.0) + (flag_weight * t_prob)
                            )
                    else:
                        border_observed_weighted[terrain] = (
                            border_observed_weighted.get(terrain, 0.0) + flag_weight
                        )
                        for t_name, t_prob in border_target_prob.items():
                            border_target_weighted[t_name] = (
                                border_target_weighted.get(t_name, 0.0) + (flag_weight * t_prob)
                            )

                report['rule_targets'].append({
                    'manager': manager.name,
                    'rule_key': rule.rule_key,
                    'province_metrics': distribution_gap_metrics(
                        observed=province_observed_weighted,
                        target=province_target_weighted,
                        domain=terrain_domain['province_terrains'],
                    ),
                    'border_metrics': distribution_gap_metrics(
                        observed=border_observed_weighted,
                        target=border_target_weighted,
                        domain=terrain_domain['border_terrains'],
                    ),
                })

        return report

    def finalize(self, total_seconds: float):
        """
        Finalize statistics at end of run.

        Args:
            total_seconds: total elapsed time for the run
        """
        if self.stats['steps'] > 0:
            self.stats['timing']['mean_step_seconds'] = total_seconds / self.stats['steps']
        self.stats['final_report'] = self.build_final_report()

    def get_statistics(self) -> dict:
        """Get the complete statistics dictionary."""
        return self.stats


# ===== DEBUG REPORT FORMATTER =====

class DebugReportFormatter:
    """
    Formats debug statistics for human-readable output.

    This class provides methods to format:
    - Progress window summaries (progress during run)
    - Progress reports (aggregated snapshot history)
    - Final reports (full distribution analysis)

    Formatting is independent of collection, allowing reuse with different data.
    """

    @staticmethod
    def format_probability_distribution_table(distribution: dict, value_label: str = "probability") -> str:
        """
        Format a probability distribution as an aligned table.

        Args:
            distribution: {terrain: probability}
            value_label: header label for the numeric column

        Returns:
            Multi-line string with a header row and aligned values
        """
        if not distribution:
            return "  (empty)"

        terrain_width = max(len("terrain"), max(len(str(terrain)) for terrain in distribution))
        value_width = max(len(value_label), 8)

        rows = [
            f"  {'terrain':<{terrain_width}}  {value_label:>{value_width}}"
        ]
        for terrain, probability in distribution.items():
            rows.append(
                f"  {terrain:<{terrain_width}}  {probability:>{value_width}.3f}"
            )

        return "\n".join(rows)

    @staticmethod
    def format_score_summary(metrics: dict, label: str, indent: str = "") -> str:
        """
        Format aggregate score metrics in a more readable compact form.

        Args:
            metrics: distribution gap metrics dict
            label: summary label to print before the metrics
            indent: optional indentation prefix

        Returns:
            One or two-line human-readable summary block
        """
        return "\n".join([
            f"{indent}{label}:",
            (
                f"{indent}  equal-weight match={metrics['equal_weight_match']:.3f} | "
                f"target-weighted match={metrics['target_weighted_match']:.3f} | "
                f"total variation distance={metrics['tv_distance']:.3f}"
            ),
        ])

    @staticmethod
    def format_per_terrain_metrics_table(per_terrain_metrics: dict) -> str:
        """
        Format per-terrain metrics as aligned table rows (one per terrain).

        Args:
            per_terrain_metrics: {terrain: {observed, target, delta, match, ...}}

        Returns:
            Multi-line string with formatted rows
        """
        if not per_terrain_metrics:
            return "  (empty)"

        terrain_width = max(len("terrain"), max(len(str(terrain)) for terrain in per_terrain_metrics))
        observed_width = max(len("observed"), 8)
        target_width = max(len("target"), 8)
        delta_width = max(len("delta"), 8)
        match_width = max(len("match"), 8)

        rows = []
        rows.append(
            f"  {'terrain':<{terrain_width}}  {'observed':>{observed_width}}  "
            f"{'target':>{target_width}}  {'delta':>{delta_width}}  {'match':>{match_width}}"
        )
        for terrain, metrics in per_terrain_metrics.items():
            row = (
                f"  {terrain:<{terrain_width}}  {metrics['observed']:>{observed_width}.3f}  "
                f"{metrics['target']:>{target_width}.3f}  {metrics['delta']:>{delta_width}.3f}  "
                f"{metrics['match']:>{match_width}.3f}"
            )
            rows.append(row)
        return "\n".join(rows)

    @staticmethod
    def format_progress_window_summary(
        summary: dict,
        include_timing: bool = False,
        include_distribution_metrics: bool = True
    ) -> str:
        """
        Format a single progress window report.

        Args:
            summary: window summary dict from collector.flush_window()
            include_timing: include timing breakdown
            include_distribution_metrics: include per-terrain metric tables

        Returns:
            Formatted multi-line string
        """
        lines = [
            "[WFC Debug Progress] "
            f"steps={summary['start_step']}-{summary['end_step']} "
            f"completion={summary['completion_ratio']:.2%} "
            f"set_provinces={summary['set_provinces']}/{summary['total_provinces']} "
            f"set_borders={summary['set_borders']}/{summary['total_borders']}"
        ]

        if include_timing:
            avg_step = summary['timing_sum_seconds'] / summary['steps'] if summary['steps'] > 0 else 0.0
            lines.append(
                f"timing_window_seconds={summary['timing_sum_seconds']:.3f} avg_step={avg_step:.3f}"
            )

        for kind in ('province', 'border'):
            kind_block = summary[kind]
            if kind_block['assignments'] <= 0:
                continue
            lines.append(f"{kind}_window_assignments={kind_block['assignments']}")
            if not include_distribution_metrics:
                lines.append(f"{kind}_observed_distribution:")
                lines.append(DebugReportFormatter.format_probability_distribution_table(
                    kind_block['observed_prob'],
                    value_label="observed",
                ))

            for rule_info in kind_block['rules']:
                metrics = rule_info['metrics']
                rule_label = f"{rule_info['manager']}::{rule_info['rule_key']}"
                if not include_distribution_metrics:
                    lines.append(f"{kind}_target_distribution[{rule_label}]:")
                    lines.append(DebugReportFormatter.format_probability_distribution_table(
                        rule_info['target_prob'],
                        value_label="target",
                    ))
                if include_distribution_metrics:
                    lines.append(
                        f"{kind}_distribution_comparison[{rule_label}]:"
                    )
                    lines.append(DebugReportFormatter.format_per_terrain_metrics_table(
                        metrics['per_terrain']
                    ))
                lines.append(DebugReportFormatter.format_score_summary(
                    metrics,
                    label=f"{kind}_scores[{rule_label}]",
                ))

        return "\n".join(lines)

    @staticmethod
    def format_progress_report(stats: dict) -> str:
        """
        Format progress report from collection statistics.

        Args:
            stats: statistics dict from collector

        Returns:
            Formatted report string
        """
        lines = [
            "WFC Progress Report",
            f"steps={stats['steps']}",
            (
                "timing_seconds="
                f"total:{stats['timing']['total_seconds']:.3f}, "
                f"province:{stats['timing']['province_assignment_seconds']:.3f}, "
                f"border:{stats['timing']['border_assignment_seconds']:.3f}, "
                f"mean_step:{stats['timing']['mean_step_seconds']:.3f}"
            ),
            f"snapshots={len(stats['progress'])}",
        ]

        if stats['progress']:
            first = stats['progress'][0]
            last = stats['progress'][-1]
            lines.append(
                "first_step="
                f"{first['step']} kind={first['element_kind']} terrain={first['selected_terrain']}"
            )
            lines.append(
                "last_step="
                f"{last['step']} kind={last['element_kind']} terrain={last['selected_terrain']} "
                f"completion={last['completion_ratio']:.2%}"
            )

        return "\n".join(lines)

    @staticmethod
    def format_final_report(
        stats: dict,
        include_distribution_metrics: bool = True
    ) -> str:
        """
        Format final distribution report.

        Args:
            stats: statistics dict from collector
            include_distribution_metrics: include per-terrain metric tables

        Returns:
            Formatted report string
        """
        final_report = stats.get('final_report', {})
        if not final_report:
            return "WFC final report is empty (run wave_function_collapse() first)."

        lines = ["WFC Final Distribution Report"]

        observed = final_report.get('observed', {})
        observed_province = observed.get('province', {})
        observed_border = observed.get('border', {})
        if not include_distribution_metrics:
            lines.append("observed_province_distribution:")
            lines.append(DebugReportFormatter.format_probability_distribution_table(
                observed_province,
                value_label="observed",
            ))
            lines.append("observed_border_distribution:")
            lines.append(DebugReportFormatter.format_probability_distribution_table(
                observed_border,
                value_label="observed",
            ))

        for rule_report in final_report.get('rule_targets', []):
            province_metrics = rule_report['province_metrics']
            border_metrics = rule_report['border_metrics']
            rule_label = f"{rule_report['manager']}::{rule_report['rule_key']}"
            lines.append(f"rule={rule_label}")
            lines.append(DebugReportFormatter.format_score_summary(
                province_metrics,
                label="province",
                indent="  ",
            ))
            lines.append(DebugReportFormatter.format_score_summary(
                border_metrics,
                label="border",
                indent="  ",
            ))

            if include_distribution_metrics:
                lines.append(f"province_distribution_comparison[{rule_label}]:")
                lines.append(DebugReportFormatter.format_per_terrain_metrics_table(
                    province_metrics['per_terrain']
                ))
                lines.append(f"border_distribution_comparison[{rule_label}]:")
                lines.append(DebugReportFormatter.format_per_terrain_metrics_table(
                    border_metrics['per_terrain']
                ))

        return "\n".join(lines)
