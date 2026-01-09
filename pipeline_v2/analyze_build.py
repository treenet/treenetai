#!/usr/bin/env python3
"""
Analyze Existing Build

Generate a comprehensive report from an existing segment build output.
This is useful when the build was run without report collection,
or to re-analyze a previous build.

Usage:
    python analyze_build.py --output-dir /path/to/processed_output
"""

import argparse
import sys
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.reporting import (
    BuildReport,
    SiteStats,
    CombinationStats,
    save_report,
    generate_text_report
)


def load_build_data(output_dir: Path) -> Dict[str, Any]:
    """Load all available data from a build output."""
    model_data_dir = output_dir / "processed" / "model_data"
    
    data = {
        'train_metadata': None,
        'test_metadata': None,
        'train_combo_ids': None,
        'test_combo_ids': None,
        'train_inputs': None,
        'test_inputs': None,
    }
    
    # Load metadata
    train_meta_path = model_data_dir / "train_metadata.pkl"
    if train_meta_path.exists():
        with open(train_meta_path, 'rb') as f:
            data['train_metadata'] = pickle.load(f)
        print(f"  Loaded train metadata: {len(data['train_metadata'])} segments")
            
    test_meta_path = model_data_dir / "test_metadata.pkl"
    if test_meta_path.exists():
        with open(test_meta_path, 'rb') as f:
            data['test_metadata'] = pickle.load(f)
        print(f"  Loaded test metadata: {len(data['test_metadata'])} segments")
            
    # Load combo IDs
    train_combo_path = model_data_dir / "train_combo_ids.pkl"
    if train_combo_path.exists():
        with open(train_combo_path, 'rb') as f:
            data['train_combo_ids'] = pickle.load(f)
        print(f"  Loaded train combo IDs: {len(data['train_combo_ids'])} combinations")
            
    test_combo_path = model_data_dir / "test_combo_ids.pkl"
    if test_combo_path.exists():
        with open(test_combo_path, 'rb') as f:
            data['test_combo_ids'] = pickle.load(f)
        print(f"  Loaded test combo IDs: {len(data['test_combo_ids'])} combinations")
            
    return data


def analyze_metadata(metadata: List, combo_ids: Dict) -> Dict[int, SiteStats]:
    """Analyze metadata to build site statistics."""
    site_stats = {}
    
    if not metadata or not combo_ids:
        return site_stats
    
    # Group segments by combo_id
    combo_segments = {}
    combo_years = {}
    combo_dates = {}
    
    for seg in metadata:
        combo_id = seg.combo_id
        if combo_id not in combo_segments:
            combo_segments[combo_id] = 0
            combo_years[combo_id] = set()
            combo_dates[combo_id] = {'start': None, 'end': None}
        
        combo_segments[combo_id] += 1
        
        # Extract year
        if hasattr(seg, 'year'):
            combo_years[combo_id].add(seg.year)
        elif hasattr(seg, 'start_date'):
            try:
                year = seg.start_date.year if hasattr(seg.start_date, 'year') else int(str(seg.start_date)[:4])
                combo_years[combo_id].add(year)
            except:
                pass
        
        # Track date range
        if hasattr(seg, 'start_date'):
            start = seg.start_date
            end = seg.end_date if hasattr(seg, 'end_date') else start
            
            if combo_dates[combo_id]['start'] is None or start < combo_dates[combo_id]['start']:
                combo_dates[combo_id]['start'] = start
            if combo_dates[combo_id]['end'] is None or end > combo_dates[combo_id]['end']:
                combo_dates[combo_id]['end'] = end
    
    # Build site stats
    for combo_id, combo_info in combo_ids.items():
        site_id = int(combo_info['site ID'])
        
        if site_id not in site_stats:
            site_stats[site_id] = SiteStats(
                site_id=site_id,
                split="unknown",
                n_thermometers=0,
                n_hygrometers=0,
                n_dendrometers=0,
                n_total_combinations=0,
                n_processed_combinations=0,
                n_successful_combinations=0,
                total_segments=0
            )
        
        site = site_stats[site_id]
        site.n_processed_combinations += 1
        
        seg_count = combo_segments.get(combo_id, 0)
        site.total_segments += seg_count
        
        if seg_count > 0:
            site.n_successful_combinations += 1
        
        # Create combination stats
        years = sorted(combo_years.get(combo_id, set()))
        dates = combo_dates.get(combo_id, {})
        
        date_start = None
        date_end = None
        if dates['start']:
            date_start = dates['start'].strftime("%Y-%m-%d") if hasattr(dates['start'], 'strftime') else str(dates['start'])
        if dates['end']:
            date_end = dates['end'].strftime("%Y-%m-%d") if hasattr(dates['end'], 'strftime') else str(dates['end'])
        
        combo_stats = CombinationStats(
            combo_id=combo_id,
            site_id=site_id,
            thermometer_id=int(combo_info['thermometer ID']),
            hygrometer_id=int(combo_info['hygrometer ID']),
            dendrometer_id=int(combo_info['dendrometer ID']),
            segment_count=seg_count,
            date_range_start=date_start,
            date_range_end=date_end,
            years_covered=years,
            total_days_covered=seg_count * 30
        )
        
        site.combinations.append(combo_stats)
        
        # Track unique sensors
        site.n_thermometers = len(set(c.thermometer_id for c in site.combinations))
        site.n_hygrometers = len(set(c.hygrometer_id for c in site.combinations))
        site.n_dendrometers = len(set(c.dendrometer_id for c in site.combinations))
    
    return site_stats


def generate_report_from_data(output_dir: str, data: Dict) -> BuildReport:
    """Generate a BuildReport from loaded data."""
    
    # Analyze train data
    train_stats = {}
    if data['train_metadata'] and data['train_combo_ids']:
        train_stats = analyze_metadata(data['train_metadata'], data['train_combo_ids'])
        for site in train_stats.values():
            site.split = 'train'
    
    # Analyze test data
    test_stats = {}
    if data['test_metadata'] and data['test_combo_ids']:
        test_stats = analyze_metadata(data['test_metadata'], data['test_combo_ids'])
        for site in test_stats.values():
            site.split = 'test'
    
    # Create report
    report = BuildReport(
        generated_at=datetime.now().isoformat(),
        output_root=str(output_dir),
        country_filter="Switzerland",  # Assume
        segment_days=30,
        stride_days=10,
        norm_scope="year",
        random_seed=42,
        max_combinations=-1,
        total_sites_available=len(train_stats) + len(test_stats),
        total_sites_processed=len(train_stats) + len(test_stats),
        train_sites_count=len(train_stats),
        test_sites_count=len(test_stats),
        total_train_segments=len(data['train_metadata']) if data['train_metadata'] else 0,
        total_test_segments=len(data['test_metadata']) if data['test_metadata'] else 0,
        train_site_stats=list(train_stats.values()),
        test_site_stats=list(test_stats.values())
    )
    
    # Calculate combination totals
    all_sites = list(train_stats.values()) + list(test_stats.values())
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


def main():
    parser = argparse.ArgumentParser(description="Analyze an existing segment build")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Path to build output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("TreeNet AI Pipeline v2 - Build Analysis")
    print("="*80)
    print(f"Analyzing: {output_dir}")
    print()
    
    print("1. Loading build data...")
    data = load_build_data(output_dir)
    
    print("\n2. Generating report...")
    report = generate_report_from_data(output_dir, data)
    
    print("\n3. Saving report...")
    report_dir = output_dir / "processed" / "model_data" / "reports"
    save_report(report, report_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    # Print summary to console
    print(f"\nSUMMARY:")
    print(f"  Total Sites: {report.total_sites_processed}")
    print(f"  Train Sites: {report.train_sites_count}")
    print(f"  Test Sites: {report.test_sites_count}")
    print(f"  Total Train Segments: {report.total_train_segments}")
    print(f"  Total Test Segments: {report.total_test_segments}")
    print(f"  Combinations Processed: {report.total_combinations_processed}")
    print(f"  Combinations Successful: {report.total_combinations_successful}")
    
    if report.year_distribution:
        print(f"\n  Years covered: {sorted(report.year_distribution.keys())}")


if __name__ == "__main__":
    main()
