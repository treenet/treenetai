"""
Visualization utilities for segment inspection.

Plots processed segments to validate data quality and coverage.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle


class SegmentPlotter:
    """
    Plot processed segments for visual inspection.
    
    Creates plots showing:
    - Input channels (11 channels, 10-min resolution)
    - Target channels (3 channels, hourly resolution)
    - Coverage statistics
    """
    
    def __init__(self, local_tz: str = 'Europe/Zurich'):
        """
        Initialize plotter.
        
        Args:
            local_tz: Local timezone for date displays
        """
        self.local_tz = local_tz
        
        self.input_channels = [
            'temp_treenet', 'rh_treenet', 'stem',
            'tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr', 'doy'
        ]
        self.target_channels = ['local_T', 'local_RH', 'stem']
        self.local_channels = ['temp_treenet', 'rh_treenet', 'stem']
        self.global_channels = ['tas', 'tasmax', 'tasmin', 'rh', 'vpd', 'gh', 'pr']
    
    def load_segments(
        self,
        data_dir: Path,
        split: str = 'train'
    ) -> tuple:
        """
        Load segment data from processed files.
        
        Args:
            data_dir: Directory with processed segment files
            split: 'train' or 'test'
            
        Returns:
            Tuple of (combo_ids, input_segments, output_segments, segment_metadata)
        """
        base_path = Path(data_dir)
        
        with open(base_path / f'model_{split}_data_combination_ids.pkl', 'rb') as f:
            combo_ids = pickle.load(f)
        
        with open(base_path / f'{split}_input_segments.pkl', 'rb') as f:
            input_segments = pickle.load(f)
        
        with open(base_path / f'{split}_output_segments.pkl', 'rb') as f:
            output_segments = pickle.load(f)
        
        with open(base_path / f'{split}_segment_ids.pkl', 'rb') as f:
            segment_metadata = pickle.load(f)
        
        return combo_ids, input_segments, output_segments, segment_metadata
    
    def plot_segment(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        combo_id: int,
        segment_idx: int,
        site_id: int,
        year: int,
        output_path: Path,
        plot_globals: bool = True
    ):
        """
        Plot a single segment.
        
        Args:
            input_df: Input DataFrame (10-min, 11 channels)
            output_df: Output DataFrame (hourly, 3 channels)
            combo_id: Combination ID
            segment_idx: Segment index
            site_id: Site ID
            year: Year
            output_path: Path to save figure
            plot_globals: Whether to overlay global channels
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2.2, 1.6])
        fig.subplots_adjust(hspace=0.28)
        
        # Convert index to day of year
        doy_input = input_df.index.tz_convert(self.local_tz).dayofyear
        doy_output = output_df.index.tz_convert(self.local_tz).dayofyear
        
        # Plot local input channels (top panel)
        colors_input = plt.cm.tab10(np.linspace(0, 1, len(self.local_channels)))
        for i, ch in enumerate(self.local_channels):
            if ch in input_df.columns:
                ax1.plot(
                    doy_input,
                    input_df[ch],
                    label=ch,
                    color=colors_input[i],
                    linewidth=1.2,
                    alpha=0.8
                )
        
        ax1.set_ylabel('Normalized (local inputs, 10-min)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9, framealpha=0.7)
        
        # Optionally plot global channels on secondary axis
        if plot_globals:
            ax1b = ax1.twinx()
            colors_global = plt.cm.tab20(np.linspace(0.5, 1, len(self.global_channels)))
            
            for i, ch in enumerate(self.global_channels):
                if ch in input_df.columns:
                    # Sample at noon for daily values
                    local_idx = input_df.index.tz_convert(self.local_tz)
                    noon_mask = (local_idx.hour == 12) & (local_idx.minute == 0)
                    if noon_mask.any():
                        noon_data = input_df.loc[noon_mask, ch]
                        noon_doy = local_idx[noon_mask].dayofyear
                        ax1b.plot(
                            noon_doy,
                            noon_data,
                            label=f'global:{ch}',
                            color=colors_global[i],
                            linewidth=1.0,
                            alpha=0.6,
                            linestyle='--'
                        )
            
            ax1b.set_ylabel('Normalized (global, daily)', fontsize=10)
            ax1b.legend(loc='upper right', fontsize=8, framealpha=0.7)
        
        # Plot target channels (bottom panel)
        colors_target = plt.cm.tab10(np.linspace(0, 1, len(self.target_channels)))
        for i, ch in enumerate(self.target_channels):
            if ch in output_df.columns:
                ax2.plot(
                    doy_output,
                    output_df[ch],
                    label=ch,
                    color=colors_target[i],
                    linewidth=1.2,
                    alpha=0.8
                )
        
        ax2.set_ylabel('Normalized (targets, hourly)', fontsize=10)
        ax2.set_xlabel('Day of Year', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9, framealpha=0.7)
        
        # Set x-axis limits
        all_doy = np.concatenate([doy_input, doy_output])
        if len(all_doy) > 0:
            doy_min, doy_max = int(all_doy.min()), int(all_doy.max())
            ax1.set_xlim(doy_min, doy_max)
            ax2.set_xlim(doy_min, doy_max)
        
        # Title
        start_date = input_df.index[0].tz_convert(self.local_tz).strftime('%Y-%m-%d')
        end_date = input_df.index[-1].tz_convert(self.local_tz).strftime('%Y-%m-%d')
        
        title = (
            f'Year {year} • Site {site_id} • Combo {combo_id} • Segment {segment_idx}\\n'
            f'{start_date} → {end_date} • '
            f'Input: {len(input_df)} steps • Output: {len(output_df)} steps'
        )
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_segments_for_site(
        self,
        data_dir: Path,
        site_id: int,
        year: int,
        output_dir: Path,
        split: str = 'train',
        max_segments: int = 10
    ):
        """
        Plot all segments for a specific site.
        
        Args:
            data_dir: Directory with processed segments
            site_id: Site ID to plot
            year: Year to plot
            output_dir: Output directory for figures
            split: 'train' or 'test'
            max_segments: Maximum number of segments to plot per site
        """
        print(f"Loading {split} segments...")
        combo_ids, input_segs, output_segs, seg_metadata = self.load_segments(
            data_dir, split
        )
        
        print(f"Finding segments for site {site_id}, year {year}...")
        
        plot_count = 0
        
        for combo_id, ids_row in combo_ids.items():
            if ids_row['site ID'] != site_id:
                continue
            
            if combo_id not in input_segs:
                continue
            
            input_seg_list = input_segs[combo_id]
            output_seg_list = output_segs[combo_id]
            
            for seg_idx, (input_df, output_df) in enumerate(zip(input_seg_list, output_seg_list)):
                if len(input_df) == 0 or len(output_df) == 0:
                    continue
                
                # Check if segment is in the target year
                seg_year = input_df.index[0].tz_convert(self.local_tz).year
                if seg_year != year:
                    continue
                
                # Plot this segment
                output_path = output_dir / f'segment_site{site_id}_combo{combo_id}_seg{seg_idx}.png'
                
                print(f"  Plotting segment: combo {combo_id}, seg {seg_idx}")
                self.plot_segment(
                    input_df=input_df,
                    output_df=output_df,
                    combo_id=combo_id,
                    segment_idx=seg_idx,
                    site_id=site_id,
                    year=year,
                    output_path=output_path
                )
                
                plot_count += 1
                
                if plot_count >= max_segments:
                    print(f"  Reached maximum of {max_segments} plots")
                    return plot_count
        
        print(f"Created {plot_count} segment plots for site {site_id}")
        return plot_count
    
    def plot_summary_stats(
        self,
        data_dir: Path,
        output_path: Path,
        split: str = 'train'
    ):
        """
        Plot summary statistics for all segments.
        
        Creates histograms of:
        - Segment lengths
        - Coverage percentages
        - Channel distributions
        
        Args:
            data_dir: Directory with processed segments
            output_path: Path to save figure
            split: 'train' or 'test'
        """
        print(f"Loading {split} segment metadata...")
        combo_ids, input_segs, output_segs, seg_metadata = self.load_segments(
            data_dir, split
        )
        
        # Collect statistics
        n_segments = []
        input_lengths = []
        output_lengths = []
        
        for combo_id in combo_ids.keys():
            if combo_id in input_segs:
                n_segments.append(len(input_segs[combo_id]))
                for input_df in input_segs[combo_id]:
                    input_lengths.append(len(input_df))
                for output_df in output_segs[combo_id]:
                    output_lengths.append(len(output_df))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Segment Statistics ({split.upper()})', fontsize=14, fontweight='bold')
        
        # Histogram of segments per combination
        axes[0, 0].hist(n_segments, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Segments per combination')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Segments per Combination')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of input lengths
        axes[0, 1].hist(input_lengths, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Input length (timesteps)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Input Segment Lengths')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of output lengths
        axes[1, 0].hist(output_lengths, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Output length (timesteps)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Output Segment Lengths')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary text
        total_combos = len(combo_ids)
        total_segments = sum(n_segments)
        avg_seg_per_combo = np.mean(n_segments) if n_segments else 0
        
        summary_text = f'''
        Total Combinations: {total_combos}
        Total Segments: {total_segments}
        Avg Segments/Combo: {avg_seg_per_combo:.1f}
        
        Input Length: {np.mean(input_lengths):.0f} ± {np.std(input_lengths):.0f}
        Output Length: {np.mean(output_lengths):.0f} ± {np.std(output_lengths):.0f}
        '''
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Summary statistics saved to: {output_path}")
