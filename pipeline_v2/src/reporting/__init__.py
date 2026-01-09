"""
Reporting module for TreeNet AI Pipeline v2.

Provides comprehensive reporting capabilities for:
- Segment building statistics
- Training metrics
- Evaluation results
- Gap analysis
"""

from .build_report import (
    BuildReport,
    BuildReportCollector,
    SiteStats,
    CombinationStats,
    GapStatistics,
    compute_gap_statistics,
    save_report,
    generate_text_report,
    analyze_existing_build,
    report_to_dict
)

__all__ = [
    'BuildReport',
    'BuildReportCollector',
    'SiteStats',
    'CombinationStats',
    'GapStatistics',
    'compute_gap_statistics',
    'save_report',
    'generate_text_report',
    'analyze_existing_build',
    'report_to_dict'
]
