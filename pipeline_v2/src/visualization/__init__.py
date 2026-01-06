"""
Visualization utilities for segment inspection.

Provides tools to:
- Plot individual segments with input and target channels
- Generate summary statistics across datasets
- Compare processed segments with raw data
"""

from .plot_segments import SegmentPlotter
from .compare_raw import RawDataComparator

__all__ = ['SegmentPlotter', 'RawDataComparator']

__all__ = ['SegmentPlotter', 'RawDataComparator']
