#!/usr/bin/env python3
"""
Build Report Generator for TreeNet AI Pipeline v2

Generates comprehensive reports on segment building process including:
- Site-level statistics
- Combination-level statistics  
- Temporal coverage analysis
- Data quality metrics
- Sensor utilization

Can be used either:
1. During build process (via BuildReportCollector)
2. Post-build analysis (via analyze_existing_build())
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np


@dataclass
class GapStatistics:
    """Statistics about gaps in time series data."""
    total_gaps: int = 0
    total_gap_timesteps: int = 0
    gap_ratio: float = 0.0  # Percentage of total timesteps that are gaps
    min_gap_length: int = 0
    max_gap_length: int = 0
    mean_gap_length: float = 0.0
    median_gap_length: float = 0.0
    gap_length_distribution: Dict[str, int] = field(default_factory=dict)  # Binned distribution


@dataclass
class CombinationStats:
    """Statistics for a single sensor combination."""
    combo_id: int
    site_id: int
    thermometer_id: int
    hygrometer_id: int
    dendrometer_id: int
    segment_count: int
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    total_days_covered: int = 0
    years_covered: List[int] = field(default_factory=list)
    input_nan_ratio: float = 0.0
    output_nan_ratio: float = 0.0
    # Gap statistics per channel
    input_gap_stats: Optional[Dict[str, GapStatistics]] = None
    output_gap_stats: Optional[Dict[str, GapStatistics]] = None


@dataclass
class SiteStats:
    """Statistics for a single site."""
    site_id: int
    site_name: str = ""  # Human-readable site name
    split: str = ""  # 'train' or 'test'
    n_thermometers: int = 0
    n_hygrometers: int = 0
    n_dendrometers: int = 0
    n_total_combinations: int = 0  # All possible combinations
    n_processed_combinations: int = 0  # Combinations with LM data
    n_successful_combinations: int = 0  # Combinations that produced segments
    total_segments: int = 0
    combinations: List[CombinationStats] = field(default_factory=list)
    has_meteo: bool = True
    status: str = "processed"  # processed, no_meteo, no_lm_data, etc.
    # Aggregated gap stats
    avg_input_gap_ratio: float = 0.0
    avg_output_gap_ratio: float = 0.0


@dataclass
class BuildReport:
    """Complete build report."""
    # Metadata
    generated_at: str
    output_root: str
    country_filter: str
    segment_days: int
    stride_days: int
    norm_scope: str
    random_seed: int
    max_combinations: int
    
    # Global stats
    total_sites_available: int = 0
    total_sites_processed: int = 0
    train_sites_count: int = 0
    test_sites_count: int = 0
    total_train_segments: int = 0
    total_test_segments: int = 0
    total_combinations_processed: int = 0
    total_combinations_successful: int = 0
    
    # Site-level data
    train_site_stats: List[SiteStats] = field(default_factory=list)
    test_site_stats: List[SiteStats] = field(default_factory=list)
    
    # Aggregated metrics
    segments_per_site_stats: Dict[str, float] = field(default_factory=dict)
    segments_per_combo_stats: Dict[str, float] = field(default_factory=dict)
    year_distribution: Dict[int, int] = field(default_factory=dict)
    sensor_utilization: Dict[str, Dict] = field(default_factory=dict)
    
    # Gap statistics summary
    global_gap_stats: Dict[str, Any] = field(default_factory=dict)
    gap_length_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)


def compute_gap_statistics(series: pd.Series, channel_name: str = "unknown") -> GapStatistics:
    """
    Compute gap statistics for a time series.
    
    A gap is defined as a consecutive sequence of NaN values.
    """
    if series is None or len(series) == 0:
        return GapStatistics()
    
    is_nan = series.isna()
    total_timesteps = len(series)
    total_nans = is_nan.sum()
    
    if total_nans == 0:
        return GapStatistics(
            total_gaps=0,
            total_gap_timesteps=0,
            gap_ratio=0.0
        )
    
    # Find gap lengths by looking at transitions
    gap_lengths = []
    in_gap = False
    current_gap_length = 0
    
    for is_nan_val in is_nan:
        if is_nan_val:
            if not in_gap:
                in_gap = True
                current_gap_length = 1
            else:
                current_gap_length += 1
        else:
            if in_gap:
                gap_lengths.append(current_gap_length)
                in_gap = False
                current_gap_length = 0
    
    # Don't forget the last gap if series ends with NaN
    if in_gap:
        gap_lengths.append(current_gap_length)
    
    if len(gap_lengths) == 0:
        return GapStatistics(
            total_gaps=0,
            total_gap_timesteps=total_nans,
            gap_ratio=100 * total_nans / total_timesteps
        )
    
    # Create binned distribution
    # Bins: 1, 2-5, 6-12, 13-24, 25-72, 73-144, 145-432, 433-1000, 1000+
    # (1 step, 2-5 steps, 6-12 steps ~1hr, 13-24 ~2-4hrs, 25-72 ~4-12hrs, 73-144 ~12-24hrs, 145-432 ~1-3 days, 433-1000 ~3-7 days, 1000+ ~week+)
    bins = {
        "1": 0,
        "2-5": 0,
        "6-12": 0,
        "13-24": 0,
        "25-72": 0,
        "73-144": 0,
        "145-432": 0,
        "433-1000": 0,
        "1000+": 0
    }
    
    for gap_len in gap_lengths:
        if gap_len == 1:
            bins["1"] += 1
        elif gap_len <= 5:
            bins["2-5"] += 1
        elif gap_len <= 12:
            bins["6-12"] += 1
        elif gap_len <= 24:
            bins["13-24"] += 1
        elif gap_len <= 72:
            bins["25-72"] += 1
        elif gap_len <= 144:
            bins["73-144"] += 1
        elif gap_len <= 432:
            bins["145-432"] += 1
        elif gap_len <= 1000:
            bins["433-1000"] += 1
        else:
            bins["1000+"] += 1
    
    return GapStatistics(
        total_gaps=len(gap_lengths),
        total_gap_timesteps=total_nans,
        gap_ratio=100 * total_nans / total_timesteps,
        min_gap_length=min(gap_lengths),
        max_gap_length=max(gap_lengths),
        mean_gap_length=np.mean(gap_lengths),
        median_gap_length=np.median(gap_lengths),
        gap_length_distribution=bins
    )


class BuildReportCollector:
    """Collects statistics during the build process."""
    
    def __init__(
        self,
        output_root: str,
        country_filter: str,
        segment_days: int = 30,
        stride_days: int = 10,
        norm_scope: str = "year",
        random_seed: int = 42,
        max_combinations: int = -1,
        metadata_path: str = "/storage/lukovic/Data/FORWARDS/treenet/server_data/metadata_all.pkl"
    ):
        """Initialize the collector."""
        self.output_root = output_root
        self.country_filter = country_filter
        self.segment_days = segment_days
        self.stride_days = stride_days
        self.norm_scope = norm_scope
        self.random_seed = random_seed
        self.max_combinations = max_combinations
        
        self.train_stats: Dict[int, SiteStats] = {}
        self.test_stats: Dict[int, SiteStats] = {}
        self.current_split = "train"
        self.combo_counter = 0
        
        # Track all sites
        self.all_available_sites: List[int] = []
        self.train_sites: List[int] = []
        self.test_sites: List[int] = []
        
        # Load site name mapping
        self.site_name_map: Dict[int, str] = {}
        self._load_site_names(metadata_path)
        
    def _load_site_names(self, metadata_path: str):
        """Load site ID to name mapping from metadata."""
        try:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                md = pd.read_pickle(metadata_path)
                site_df = md[['site_id', 'site_name']].drop_duplicates(subset='site_id')
                self.site_name_map = dict(zip(site_df['site_id'].astype(int), site_df['site_name']))
                print(f"  Loaded {len(self.site_name_map)} site names from metadata")
        except Exception as e:
            print(f"  Warning: Could not load site names: {e}")
        
    def set_available_sites(self, sites: List[int]):
        """Record all available sites."""
        self.all_available_sites = sorted(sites)
        
    def set_train_test_split(self, train_sites: List[int], test_sites: List[int]):
        """Record train/test split."""
        self.train_sites = sorted(train_sites)
        self.test_sites = sorted(test_sites)
        
    def set_current_split(self, split: str):
        """Set current processing split."""
        self.current_split = split
        
    def start_site(
        self,
        site_id: int,
        n_thermometers: int,
        n_hygrometers: int,
        n_dendrometers: int,
        has_meteo: bool = True
    ):
        """Start recording for a new site."""
        site_name = self.site_name_map.get(int(site_id), f"Site_{site_id}")
        
        stats = SiteStats(
            site_id=site_id,
            site_name=site_name,
            split=self.current_split,
            n_thermometers=n_thermometers,
            n_hygrometers=n_hygrometers,
            n_dendrometers=n_dendrometers,
            n_total_combinations=n_thermometers * n_hygrometers * n_dendrometers,
            n_processed_combinations=0,
            n_successful_combinations=0,
            total_segments=0,
            has_meteo=has_meteo,
            status="processing" if has_meteo else "no_meteo"
        )
        
        if self.current_split == "train":
            self.train_stats[site_id] = stats
        else:
            self.test_stats[site_id] = stats
            
    def record_combination(
        self,
        site_id: int,
        thermometer_id: int,
        hygrometer_id: int,
        dendrometer_id: int,
        segment_count: int,
        segment_metadata: List[Any] = None,
        input_nan_ratio: float = 0.0,
        output_nan_ratio: float = 0.0,
        input_df: pd.DataFrame = None,
        output_df: pd.DataFrame = None
    ):
        """Record statistics for a processed combination."""
        self.combo_counter += 1
        
        stats_dict = self.train_stats if self.current_split == "train" else self.test_stats
        site_stats = stats_dict.get(site_id)
        
        if site_stats is None:
            return
            
        site_stats.n_processed_combinations += 1
        
        # Extract temporal info from segment metadata
        date_start = None
        date_end = None
        years = []
        total_days = 0
        
        if segment_metadata and len(segment_metadata) > 0:
            try:
                starts = [m.start_date for m in segment_metadata if hasattr(m, 'start_date')]
                ends = [m.end_date for m in segment_metadata if hasattr(m, 'end_date')]
                if starts:
                    date_start = min(starts).strftime("%Y-%m-%d") if hasattr(min(starts), 'strftime') else str(min(starts))
                if ends:
                    date_end = max(ends).strftime("%Y-%m-%d") if hasattr(max(ends), 'strftime') else str(max(ends))
                
                # Extract years
                years_set = set()
                for m in segment_metadata:
                    if hasattr(m, 'year'):
                        years_set.add(m.year)
                    elif hasattr(m, 'start_date'):
                        years_set.add(m.start_date.year if hasattr(m.start_date, 'year') else int(str(m.start_date)[:4]))
                years = sorted(years_set)
                
                # Calculate total days covered
                total_days = len(segment_metadata) * self.segment_days
            except Exception:
                pass
        
        # Compute gap statistics if dataframes provided
        input_gap_stats = None
        output_gap_stats = None
        
        if input_df is not None:
            input_gap_stats = {}
            for col in input_df.columns:
                input_gap_stats[col] = compute_gap_statistics(input_df[col], col)
            # Compute overall input NaN ratio
            total_vals = input_df.size
            total_nans = input_df.isna().sum().sum()
            input_nan_ratio = 100 * total_nans / total_vals if total_vals > 0 else 0
        
        if output_df is not None:
            output_gap_stats = {}
            for col in output_df.columns:
                output_gap_stats[col] = compute_gap_statistics(output_df[col], col)
            # Compute overall output NaN ratio
            total_vals = output_df.size
            total_nans = output_df.isna().sum().sum()
            output_nan_ratio = 100 * total_nans / total_vals if total_vals > 0 else 0
        
        combo_stats = CombinationStats(
            combo_id=self.combo_counter,
            site_id=site_id,
            thermometer_id=thermometer_id,
            hygrometer_id=hygrometer_id,
            dendrometer_id=dendrometer_id,
            segment_count=segment_count,
            date_range_start=date_start,
            date_range_end=date_end,
            total_days_covered=total_days,
            years_covered=years,
            input_nan_ratio=input_nan_ratio,
            output_nan_ratio=output_nan_ratio,
            input_gap_stats=input_gap_stats,
            output_gap_stats=output_gap_stats
        )
        
        site_stats.combinations.append(combo_stats)
        site_stats.total_segments += segment_count
        
        if segment_count > 0:
            site_stats.n_successful_combinations += 1
            
    def finalize_site(self, site_id: int):
        """Finalize statistics for a site."""
        stats_dict = self.train_stats if self.current_split == "train" else self.test_stats
        site_stats = stats_dict.get(site_id)
        
        if site_stats and site_stats.status == "processing":
            if site_stats.total_segments > 0:
                site_stats.status = "processed"
            elif site_stats.n_processed_combinations == 0:
                site_stats.status = "no_lm_data"
            else:
                site_stats.status = "no_valid_segments"
                
    def generate_report(self) -> BuildReport:
        """Generate the final report."""
        report = BuildReport(
            generated_at=datetime.now().isoformat(),
            output_root=self.output_root,
            country_filter=self.country_filter,
            segment_days=self.segment_days,
            stride_days=self.stride_days,
            norm_scope=self.norm_scope,
            random_seed=self.random_seed,
            max_combinations=self.max_combinations,
            total_sites_available=len(self.all_available_sites),
            total_sites_processed=len(self.train_stats) + len(self.test_stats),
            train_sites_count=len(self.train_sites),
            test_sites_count=len(self.test_sites),
        )
        
        # Collect all site stats
        report.train_site_stats = list(self.train_stats.values())
        report.test_site_stats = list(self.test_stats.values())
        
        # Calculate totals
        report.total_train_segments = sum(s.total_segments for s in report.train_site_stats)
        report.total_test_segments = sum(s.total_segments for s in report.test_site_stats)
        
        all_sites = report.train_site_stats + report.test_site_stats
        report.total_combinations_processed = sum(s.n_processed_combinations for s in all_sites)
        report.total_combinations_successful = sum(s.n_successful_combinations for s in all_sites)
        
        # Calculate per-site stats
        segments_per_site = [s.total_segments for s in all_sites if s.total_segments > 0]
        if segments_per_site:
            report.segments_per_site_stats = {
                "min": float(np.min(segments_per_site)),
                "max": float(np.max(segments_per_site)),
                "mean": float(np.mean(segments_per_site)),
                "median": float(np.median(segments_per_site)),
                "std": float(np.std(segments_per_site)),
                "total": float(sum(segments_per_site))
            }
            
        # Calculate per-combo stats
        all_combo_segments = []
        for site in all_sites:
            for combo in site.combinations:
                if combo.segment_count > 0:
                    all_combo_segments.append(combo.segment_count)
                    
        if all_combo_segments:
            report.segments_per_combo_stats = {
                "min": float(np.min(all_combo_segments)),
                "max": float(np.max(all_combo_segments)),
                "mean": float(np.mean(all_combo_segments)),
                "median": float(np.median(all_combo_segments)),
                "std": float(np.std(all_combo_segments)),
                "total": float(sum(all_combo_segments))
            }
            
        # Year distribution
        year_counts = {}
        for site in all_sites:
            for combo in site.combinations:
                for year in combo.years_covered:
                    year_counts[year] = year_counts.get(year, 0) + combo.segment_count
        report.year_distribution = year_counts
        
        # Sensor utilization
        used_thermometers = set()
        used_hygrometers = set()
        used_dendrometers = set()
        total_thermometers = set()
        total_hygrometers = set()
        total_dendrometers = set()
        
        for site in all_sites:
            for combo in site.combinations:
                total_thermometers.add((site.site_id, combo.thermometer_id))
                total_hygrometers.add((site.site_id, combo.hygrometer_id))
                total_dendrometers.add((site.site_id, combo.dendrometer_id))
                if combo.segment_count > 0:
                    used_thermometers.add((site.site_id, combo.thermometer_id))
                    used_hygrometers.add((site.site_id, combo.hygrometer_id))
                    used_dendrometers.add((site.site_id, combo.dendrometer_id))
                    
        report.sensor_utilization = {
            "thermometers": {
                "total": len(total_thermometers),
                "used": len(used_thermometers),
                "utilization_pct": 100 * len(used_thermometers) / max(1, len(total_thermometers))
            },
            "hygrometers": {
                "total": len(total_hygrometers),
                "used": len(used_hygrometers),
                "utilization_pct": 100 * len(used_hygrometers) / max(1, len(total_hygrometers))
            },
            "dendrometers": {
                "total": len(total_dendrometers),
                "used": len(used_dendrometers),
                "utilization_pct": 100 * len(used_dendrometers) / max(1, len(total_dendrometers))
            }
        }
        
        return report


def report_to_dict(report: BuildReport) -> dict:
    """Convert BuildReport to dictionary for JSON serialization."""
    def convert(obj):
        if isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif hasattr(obj, '__dataclass_fields__'):
            return {k: convert(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    return convert(report)


def save_report(report: BuildReport, output_dir: Path):
    """Save report in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_dict = report_to_dict(report)
    
    # Save JSON report
    json_path = output_dir / "build_report.json"
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    print(f"  Saved JSON report: {json_path}")
    
    # Save pickle for full Python access
    pickle_path = output_dir / "build_report.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(report, f)
    print(f"  Saved pickle report: {pickle_path}")
    
    # Save human-readable text report
    text_path = output_dir / "build_report.txt"
    with open(text_path, 'w') as f:
        f.write(generate_text_report(report))
    print(f"  Saved text report: {text_path}")
    
    # Save CSV summaries
    save_csv_summaries(report, output_dir)
    
    return json_path, pickle_path, text_path


def generate_text_report(report: BuildReport) -> str:
    """Generate a human-readable text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("TREENET AI PIPELINE v2 - SEGMENT BUILDING REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {report.generated_at}")
    lines.append(f"Output Directory: {report.output_root}")
    lines.append("")
    
    # Configuration
    lines.append("-" * 80)
    lines.append("CONFIGURATION")
    lines.append("-" * 80)
    lines.append(f"  Country Filter: {report.country_filter}")
    lines.append(f"  Segment Length: {report.segment_days} days")
    lines.append(f"  Stride: {report.stride_days} days")
    lines.append(f"  Normalization: {report.norm_scope}")
    lines.append(f"  Random Seed: {report.random_seed}")
    lines.append(f"  Max Combinations: {'All' if report.max_combinations < 0 else report.max_combinations}")
    lines.append("")
    
    # Global Summary
    lines.append("-" * 80)
    lines.append("GLOBAL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  Sites Available: {report.total_sites_available}")
    lines.append(f"  Sites Processed: {report.total_sites_processed}")
    lines.append(f"    - Train Sites: {report.train_sites_count}")
    lines.append(f"    - Test Sites: {report.test_sites_count}")
    lines.append("")
    lines.append(f"  Total Segments: {report.total_train_segments + report.total_test_segments}")
    lines.append(f"    - Train Segments: {report.total_train_segments}")
    lines.append(f"    - Test Segments: {report.total_test_segments}")
    lines.append("")
    lines.append(f"  Combinations Processed: {report.total_combinations_processed}")
    lines.append(f"  Combinations Successful: {report.total_combinations_successful} ({100*report.total_combinations_successful/max(1, report.total_combinations_processed):.1f}%)")
    lines.append("")
    
    # Segment Statistics
    if report.segments_per_site_stats:
        lines.append("-" * 80)
        lines.append("SEGMENTS PER SITE (sites with >0 segments)")
        lines.append("-" * 80)
        stats = report.segments_per_site_stats
        lines.append(f"  Min: {stats.get('min', 0):.0f}")
        lines.append(f"  Max: {stats.get('max', 0):.0f}")
        lines.append(f"  Mean: {stats.get('mean', 0):.1f}")
        lines.append(f"  Median: {stats.get('median', 0):.1f}")
        lines.append(f"  Std Dev: {stats.get('std', 0):.1f}")
        lines.append("")
        
    if report.segments_per_combo_stats:
        lines.append("-" * 80)
        lines.append("SEGMENTS PER COMBINATION (combos with >0 segments)")
        lines.append("-" * 80)
        stats = report.segments_per_combo_stats
        lines.append(f"  Min: {stats.get('min', 0):.0f}")
        lines.append(f"  Max: {stats.get('max', 0):.0f}")
        lines.append(f"  Mean: {stats.get('mean', 0):.1f}")
        lines.append(f"  Median: {stats.get('median', 0):.1f}")
        lines.append(f"  Std Dev: {stats.get('std', 0):.1f}")
        lines.append("")
        
    # Year Distribution
    if report.year_distribution:
        lines.append("-" * 80)
        lines.append("YEAR DISTRIBUTION (segments per year)")
        lines.append("-" * 80)
        for year in sorted(report.year_distribution.keys()):
            count = report.year_distribution[year]
            bar = "█" * int(count / max(report.year_distribution.values()) * 40)
            lines.append(f"  {year}: {count:6d} {bar}")
        lines.append("")
        
    # Sensor Utilization
    if report.sensor_utilization:
        lines.append("-" * 80)
        lines.append("SENSOR UTILIZATION")
        lines.append("-" * 80)
        for sensor_type, stats in report.sensor_utilization.items():
            lines.append(f"  {sensor_type.capitalize()}:")
            lines.append(f"    Total: {stats['total']}, Used: {stats['used']} ({stats['utilization_pct']:.1f}%)")
        lines.append("")
        
    # Train Sites Detail
    lines.append("-" * 80)
    lines.append("TRAIN SITES DETAIL")
    lines.append("-" * 80)
    lines.append(f"{'ID':>5} {'Site Name':<25} {'Status':<12} {'T':>2} {'H':>2} {'D':>2} {'Combos':>7} {'OK':>5} {'Segments':>9}")
    lines.append("-" * 80)
    
    for site in sorted(report.train_site_stats, key=lambda x: x.site_id):
        name = site.site_name[:24] if site.site_name else f"Site_{site.site_id}"
        lines.append(
            f"{site.site_id:>5} {name:<25} {site.status:<12} {site.n_thermometers:>2} {site.n_hygrometers:>2} "
            f"{site.n_dendrometers:>2} {site.n_processed_combinations:>7} {site.n_successful_combinations:>5} "
            f"{site.total_segments:>9}"
        )
    lines.append("")
    
    # Test Sites Detail
    lines.append("-" * 80)
    lines.append("TEST SITES DETAIL")
    lines.append("-" * 80)
    lines.append(f"{'ID':>5} {'Site Name':<25} {'Status':<12} {'T':>2} {'H':>2} {'D':>2} {'Combos':>7} {'OK':>5} {'Segments':>9}")
    lines.append("-" * 80)
    
    for site in sorted(report.test_site_stats, key=lambda x: x.site_id):
        name = site.site_name[:24] if site.site_name else f"Site_{site.site_id}"
        lines.append(
            f"{site.site_id:>5} {name:<25} {site.status:<12} {site.n_thermometers:>2} {site.n_hygrometers:>2} "
            f"{site.n_dendrometers:>2} {site.n_processed_combinations:>7} {site.n_successful_combinations:>5} "
            f"{site.total_segments:>9}"
        )
    lines.append("")
    
    # Top Producing Combinations
    lines.append("-" * 80)
    lines.append("TOP 20 PRODUCING COMBINATIONS")
    lines.append("-" * 80)
    
    all_combos = []
    for site in report.train_site_stats + report.test_site_stats:
        for combo in site.combinations:
            all_combos.append(combo)
            
    top_combos = sorted(all_combos, key=lambda x: x.segment_count, reverse=True)[:20]
    lines.append(f"{'Combo':>6} {'Site':>6} {'T':>5} {'H':>5} {'D':>5} {'Segments':>10} {'Date Range':<25}")
    lines.append("-" * 80)
    
    for combo in top_combos:
        date_range = f"{combo.date_range_start or '?'} to {combo.date_range_end or '?'}"
        lines.append(
            f"{combo.combo_id:>6} {combo.site_id:>6} {combo.thermometer_id:>5} {combo.hygrometer_id:>5} "
            f"{combo.dendrometer_id:>5} {combo.segment_count:>10} {date_range:<25}"
        )
    lines.append("")
    
    # Sites with No Segments
    zero_sites_train = [s for s in report.train_site_stats if s.total_segments == 0]
    zero_sites_test = [s for s in report.test_site_stats if s.total_segments == 0]
    
    if zero_sites_train or zero_sites_test:
        lines.append("-" * 80)
        lines.append("SITES WITH NO SEGMENTS (need investigation)")
        lines.append("-" * 80)
        
        for site in zero_sites_train + zero_sites_test:
            name = site.site_name if site.site_name else f"Site_{site.site_id}"
            lines.append(f"  Site {site.site_id} - {name} ({site.split}): {site.status}")
            lines.append(f"    Sensors: T={site.n_thermometers}, H={site.n_hygrometers}, D={site.n_dendrometers}")
            lines.append(f"    Possible combinations: {site.n_total_combinations}")
            lines.append(f"    Processed: {site.n_processed_combinations}")
        lines.append("")
    
    # Gap Statistics Summary (if available)
    has_gap_stats = False
    all_gap_ratios_input = []
    all_gap_ratios_output = []
    gap_length_totals = {
        "1": 0, "2-5": 0, "6-12": 0, "13-24": 0, 
        "25-72": 0, "73-144": 0, "145-432": 0, "433-1000": 0, "1000+": 0
    }
    
    for site in report.train_site_stats + report.test_site_stats:
        for combo in site.combinations:
            if combo.input_gap_stats:
                has_gap_stats = True
                for channel, stats in combo.input_gap_stats.items():
                    if hasattr(stats, 'gap_ratio'):
                        all_gap_ratios_input.append(stats.gap_ratio)
                    if hasattr(stats, 'gap_length_distribution'):
                        for bin_name, count in stats.gap_length_distribution.items():
                            if bin_name in gap_length_totals:
                                gap_length_totals[bin_name] += count
            if combo.output_gap_stats:
                for channel, stats in combo.output_gap_stats.items():
                    if hasattr(stats, 'gap_ratio'):
                        all_gap_ratios_output.append(stats.gap_ratio)
    
    if has_gap_stats and (all_gap_ratios_input or all_gap_ratios_output):
        lines.append("-" * 80)
        lines.append("GAP STATISTICS SUMMARY")
        lines.append("-" * 80)
        
        if all_gap_ratios_input:
            lines.append("  Input Data Gap Ratios (%):")
            lines.append(f"    Mean: {np.mean(all_gap_ratios_input):.2f}%")
            lines.append(f"    Median: {np.median(all_gap_ratios_input):.2f}%")
            lines.append(f"    Max: {np.max(all_gap_ratios_input):.2f}%")
        
        if all_gap_ratios_output:
            lines.append("  Output Data Gap Ratios (%):")
            lines.append(f"    Mean: {np.mean(all_gap_ratios_output):.2f}%")
            lines.append(f"    Median: {np.median(all_gap_ratios_output):.2f}%")
            lines.append(f"    Max: {np.max(all_gap_ratios_output):.2f}%")
        
        lines.append("")
        lines.append("  Gap Length Distribution (all input channels):")
        lines.append("  (For 10-min data: 6 steps = 1hr, 144 = 1 day, 1008 = 1 week)")
        total_gaps = sum(gap_length_totals.values())
        if total_gaps > 0:
            for bin_name in ["1", "2-5", "6-12", "13-24", "25-72", "73-144", "145-432", "433-1000", "1000+"]:
                count = gap_length_totals[bin_name]
                pct = 100 * count / total_gaps
                bar = "█" * int(pct / 2.5)
                lines.append(f"    {bin_name:>10}: {count:8d} ({pct:5.1f}%) {bar}")
        lines.append("")
        
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def save_csv_summaries(report: BuildReport, output_dir: Path):
    """Save detailed CSV summaries for further analysis."""
    output_dir = Path(output_dir)
    
    # Site-level summary
    site_data = []
    for site in report.train_site_stats + report.test_site_stats:
        site_data.append({
            'site_id': site.site_id,
            'site_name': site.site_name,
            'split': site.split,
            'status': site.status,
            'n_thermometers': site.n_thermometers,
            'n_hygrometers': site.n_hygrometers,
            'n_dendrometers': site.n_dendrometers,
            'n_total_combinations': site.n_total_combinations,
            'n_processed_combinations': site.n_processed_combinations,
            'n_successful_combinations': site.n_successful_combinations,
            'total_segments': site.total_segments,
            'has_meteo': site.has_meteo,
            'avg_input_gap_ratio': site.avg_input_gap_ratio,
            'avg_output_gap_ratio': site.avg_output_gap_ratio
        })
    
    site_df = pd.DataFrame(site_data)
    site_path = output_dir / "site_summary.csv"
    site_df.to_csv(site_path, index=False)
    print(f"  Saved site summary: {site_path}")
    
    # Combination-level summary
    combo_data = []
    for site in report.train_site_stats + report.test_site_stats:
        for combo in site.combinations:
            row = {
                'combo_id': combo.combo_id,
                'site_id': combo.site_id,
                'site_name': site.site_name,
                'split': site.split,
                'thermometer_id': combo.thermometer_id,
                'hygrometer_id': combo.hygrometer_id,
                'dendrometer_id': combo.dendrometer_id,
                'segment_count': combo.segment_count,
                'date_range_start': combo.date_range_start,
                'date_range_end': combo.date_range_end,
                'total_days_covered': combo.total_days_covered,
                'years_covered': ','.join(map(str, combo.years_covered)),
                'input_nan_ratio': combo.input_nan_ratio,
                'output_nan_ratio': combo.output_nan_ratio
            }
            
            # Add per-channel gap stats if available
            if combo.input_gap_stats:
                for ch, stats in combo.input_gap_stats.items():
                    if hasattr(stats, 'gap_ratio'):
                        row[f'gap_ratio_{ch}'] = stats.gap_ratio
                    if hasattr(stats, 'total_gaps'):
                        row[f'num_gaps_{ch}'] = stats.total_gaps
            
            combo_data.append(row)
            
    combo_df = pd.DataFrame(combo_data)
    combo_path = output_dir / "combination_summary.csv"
    combo_df.to_csv(combo_path, index=False)
    print(f"  Saved combination summary: {combo_path}")


def analyze_existing_build(output_dir: str) -> BuildReport:
    """
    Analyze an existing build output to generate a report.
    
    This can be used when the build was run without report collection,
    by reading the saved segment files.
    """
    output_dir = Path(output_dir) / "processed" / "model_data"
    
    report = BuildReport(
        generated_at=datetime.now().isoformat(),
        output_root=str(output_dir.parent.parent),
        country_filter="unknown",
        segment_days=30,
        stride_days=10,
        norm_scope="year",
        random_seed=42,
        max_combinations=-1
    )
    
    # Load train metadata
    train_meta_path = output_dir / "train_metadata.pkl"
    if train_meta_path.exists():
        with open(train_meta_path, 'rb') as f:
            train_metadata = pickle.load(f)
        report.total_train_segments = len(train_metadata)
        
    # Load test metadata
    test_meta_path = output_dir / "test_metadata.pkl"
    if test_meta_path.exists():
        with open(test_meta_path, 'rb') as f:
            test_metadata = pickle.load(f)
        report.total_test_segments = len(test_metadata)
        
    # Load combo IDs
    train_combo_path = output_dir / "train_combo_ids.pkl"
    test_combo_path = output_dir / "test_combo_ids.pkl"
    
    if train_combo_path.exists():
        with open(train_combo_path, 'rb') as f:
            train_combos = pickle.load(f)
        report.train_sites_count = len(set(c['site ID'] for c in train_combos.values()))
        
    if test_combo_path.exists():
        with open(test_combo_path, 'rb') as f:
            test_combos = pickle.load(f)
        report.test_sites_count = len(set(c['site ID'] for c in test_combos.values()))
        
    report.total_sites_processed = report.train_sites_count + report.test_sites_count
    report.total_sites_available = report.total_sites_processed
    
    return report


if __name__ == "__main__":
    # Test with existing build
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate build report")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Path to build output directory")
    args = parser.parse_args()
    
    print("Analyzing existing build...")
    report = analyze_existing_build(args.output_dir)
    
    print("\nSaving report...")
    save_report(report, Path(args.output_dir) / "reports")
    print("\nDone!")
