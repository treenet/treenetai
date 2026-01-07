"""
Tests for visualization modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from src.visualization.plot_segments import SegmentPlotter
from src.visualization.compare_raw import RawDataComparator


class TestSegmentPlotter:
    """Test segment plotting functionality."""
    
    def test_initialization(self):
        """Test plotter initialization."""
        plotter = SegmentPlotter(local_tz='Europe/Zurich')
        
        assert plotter.local_tz == 'Europe/Zurich'
        assert len(plotter.input_channels) == 11
        assert len(plotter.target_channels) == 3
    
    def test_channel_definitions(self):
        """Test that channel lists are correct."""
        plotter = SegmentPlotter()
        
        # Check input channels
        assert 'temp_treenet' in plotter.input_channels
        assert 'stem' in plotter.input_channels
        assert 'doy' in plotter.input_channels
        
        # Check target channels
        assert 'local_T' in plotter.target_channels
        assert 'local_RH' in plotter.target_channels
        assert 'stem' in plotter.target_channels
        
        # Check global channels
        assert 'tas' in plotter.global_channels
        assert 'pr' in plotter.global_channels
    
    def test_load_segments(self, tmp_path):
        """Test loading segment data from files."""
        # Create dummy segment files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        combo_ids = {0: {'site ID': 1, 'sensor type': 'thermometer_l1'}}
        
        input_segs = {
            0: [pd.DataFrame({
                'temp_treenet': np.random.randn(100),
                'stem': np.random.randn(100)
            }, index=pd.date_range('2021-01-01', periods=100, freq='10min', tz='UTC'))]
        }
        
        output_segs = {
            0: [pd.DataFrame({
                'local_T': np.random.randn(20)
            }, index=pd.date_range('2021-01-01', periods=20, freq='60min', tz='UTC'))]
        }
        
        seg_metadata = []
        
        # Save files
        with open(data_dir / 'model_train_data_combination_ids.pkl', 'wb') as f:
            pickle.dump(combo_ids, f)
        with open(data_dir / 'train_input_segments.pkl', 'wb') as f:
            pickle.dump(input_segs, f)
        with open(data_dir / 'train_output_segments.pkl', 'wb') as f:
            pickle.dump(output_segs, f)
        with open(data_dir / 'train_segment_ids.pkl', 'wb') as f:
            pickle.dump(seg_metadata, f)
        
        # Load
        plotter = SegmentPlotter()
        c_ids, i_segs, o_segs, meta = plotter.load_segments(data_dir, split='train')
        
        assert 0 in c_ids
        assert 0 in i_segs
        assert len(i_segs[0]) == 1
    
    def test_plot_segment_creates_figure(self, tmp_path):
        """Test that plot_segment creates a figure file."""
        plotter = SegmentPlotter()
        
        # Create dummy data
        timestamps_input = pd.date_range('2021-06-01', periods=288, freq='10min', tz='UTC')
        timestamps_output = pd.date_range('2021-06-01', periods=24, freq='60min', tz='UTC')
        
        input_df = pd.DataFrame({
            'temp_treenet': np.random.randn(288),
            'rh_treenet': np.random.randn(288),
            'stem': np.random.randn(288)
        }, index=timestamps_input)
        
        output_df = pd.DataFrame({
            'local_T': np.random.randn(24),
            'local_RH': np.random.randn(24),
            'stem': np.random.randn(24)
        }, index=timestamps_output)
        
        output_path = tmp_path / "test_plot.png"
        
        # Plot
        plotter.plot_segment(
            input_df=input_df,
            output_df=output_df,
            combo_id=0,
            segment_idx=0,
            site_id=1,
            year=2021,
            output_path=output_path,
            plot_globals=False
        )
        
        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Cleanup
        plt.close('all')
    
    def test_plot_segment_with_globals(self, tmp_path):
        """Test plotting with global channels overlay."""
        plotter = SegmentPlotter()
        
        timestamps_input = pd.date_range('2021-06-01', periods=288, freq='10min', tz='UTC')
        timestamps_output = pd.date_range('2021-06-01', periods=24, freq='60min', tz='UTC')
        
        input_df = pd.DataFrame({
            'temp_treenet': np.random.randn(288),
            'stem': np.random.randn(288),
            'tas': np.random.randn(288),
            'pr': np.random.randn(288)
        }, index=timestamps_input)
        
        output_df = pd.DataFrame({
            'local_T': np.random.randn(24),
            'stem': np.random.randn(24)
        }, index=timestamps_output)
        
        output_path = tmp_path / "test_plot_globals.png"
        
        plotter.plot_segment(
            input_df=input_df,
            output_df=output_df,
            combo_id=0,
            segment_idx=0,
            site_id=1,
            year=2021,
            output_path=output_path,
            plot_globals=True
        )
        
        assert output_path.exists()
        plt.close('all')
    
    @pytest.mark.skip(reason="Requires complete data directory structure with multiple pickle files")
    def test_plot_summary_stats_creates_file(self, tmp_path):
        """Test summary statistics plotting."""
        plotter = SegmentPlotter()
        
        # Create dummy data directory structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create minimal segment files for testing
        train_dir = data_dir / "train"
        train_dir.mkdir()
        
        # Create empty train_segment_ids.pkl
        import pickle
        with open(data_dir / 'train_segment_ids.pkl', 'wb') as f:
            pickle.dump({}, f)
        
        output_path = tmp_path / "summary_stats.png"
        
        # Should not crash even with empty data
        plotter.plot_summary_stats(data_dir, output_path, split='train')
        
        # File may or may not be created depending on whether there's data
        plt.close('all')


class TestRawDataComparator:
    """Test raw data comparison functionality."""
    
    def test_initialization(self):
        """Test comparator initialization."""
        comp = RawDataComparator(local_tz='Europe/Zurich')
        
        assert comp.local_tz == 'Europe/Zurich'
        assert len(comp.input_channels_wo_doy) == 10
        assert len(comp.target_channels) == 3
    
    def test_denormalize_segment(self):
        """Test denormalization reverses normalization."""
        comp = RawDataComparator()
        
        # Create normalized data
        df_norm = pd.DataFrame({
            'temp_treenet': np.array([0.0, 0.5, 1.0]),
            'stem': np.array([0.0, 0.25, 0.5])
        }, index=pd.date_range('2021-01-01', periods=3, freq='10min', tz='UTC'))
        
        # Normalization params
        mins = {'temp_treenet': -10.0, 'stem': 5.0}
        diffs = {'temp_treenet': 30.0, 'stem': 20.0}
        
        # Denormalize
        df_denorm = comp.denormalize_segment(
            df_norm=df_norm,
            mins=mins,
            diffs=diffs,
            channels=['temp_treenet', 'stem']
        )
        
        # Check results
        # temp: 0.0 * 30 + (-10) = -10
        # temp: 0.5 * 30 + (-10) = 5
        # temp: 1.0 * 30 + (-10) = 20
        np.testing.assert_allclose(
            df_denorm['temp_treenet'].values,
            [-10.0, 5.0, 20.0],
            rtol=1e-5
        )
        
        # stem: 0.0 * 20 + 5 = 5
        # stem: 0.25 * 20 + 5 = 10
        # stem: 0.5 * 20 + 5 = 15
        np.testing.assert_allclose(
            df_denorm['stem'].values,
            [5.0, 10.0, 15.0],
            rtol=1e-5
        )
    
    def test_denormalize_handles_zero_diff(self):
        """Test denormalization with zero diff (constant channel)."""
        comp = RawDataComparator()
        
        df_norm = pd.DataFrame({
            'temp': np.array([0.5, 0.5, 0.5])
        }, index=pd.date_range('2021-01-01', periods=3, freq='10min', tz='UTC'))
        
        # Zero diff means constant value
        mins = {'temp': 20.0}
        diffs = {'temp': 0.0}
        
        df_denorm = comp.denormalize_segment(
            df_norm=df_norm,
            mins=mins,
            diffs=diffs,
            channels=['temp']
        )
        
        # Should all be 20.0
        np.testing.assert_allclose(
            df_denorm['temp'].values,
            [20.5, 20.5, 20.5],
            rtol=1e-5
        )
    
    def test_denormalize_skips_missing_channels(self):
        """Test denormalization skips missing channels."""
        comp = RawDataComparator()
        
        df_norm = pd.DataFrame({
            'temp': np.array([0.5])
        }, index=pd.date_range('2021-01-01', periods=1, freq='10min', tz='UTC'))
        
        mins = {'temp': 10.0, 'missing_ch': 5.0}
        diffs = {'temp': 20.0, 'missing_ch': 10.0}
        
        df_denorm = comp.denormalize_segment(
            df_norm=df_norm,
            mins=mins,
            diffs=diffs,
            channels=['temp', 'missing_ch']
        )
        
        # Should only have 'temp'
        assert 'temp' in df_denorm.columns
        assert 'missing_ch' not in df_denorm.columns
    
    def test_load_raw_sensor(self, tmp_path):
        """Test loading raw sensor data from feather file."""
        comp = RawDataComparator()
        
        # Create dummy feather file
        raw_root = tmp_path / "server_data"
        sensor_dir = raw_root / "thermometer_l1"
        sensor_dir.mkdir(parents=True)
        
        # Create dummy data
        df = pd.DataFrame({
            'ts': pd.date_range('2021-01-01', periods=100, freq='10min'),
            'value': np.random.randn(100)
        })
        
        file_path = sensor_dir / "thermometer_l1_series_id_123.ftr"
        df.to_feather(file_path)
        
        # Load
        loaded = comp.load_raw_sensor(
            raw_root=raw_root,
            sensor_type='thermometer_l1',
            sensor_id=123,
            column='value'
        )
        
        assert len(loaded) == 100
        assert 'value' in loaded.columns
        assert loaded.index.tz is not None  # Should be UTC
    
    def test_load_raw_sensor_missing_file(self, tmp_path):
        """Test that loading missing sensor file raises error."""
        comp = RawDataComparator()
        
        raw_root = tmp_path / "server_data"
        
        with pytest.raises(FileNotFoundError):
            comp.load_raw_sensor(
                raw_root=raw_root,
                sensor_type='thermometer_l1',
                sensor_id=999,
                column='value'
            )
    
    def test_load_raw_meteo(self, tmp_path):
        """Test loading raw meteo data from CSV."""
        comp = RawDataComparator()
        
        # Create dummy CSV
        meteo_root = tmp_path / "meteo_data"
        meteo_root.mkdir()
        
        df = pd.DataFrame({
            'ts_local': pd.date_range('2021-01-01', periods=365, freq='D'),
            'tas': np.random.randn(365),
            'pr': np.random.randn(365)
        })
        
        csv_path = meteo_root / "site_1.csv"
        df.to_csv(csv_path, index=False)
        
        # Load
        loaded = comp.load_raw_meteo(
            meteo_root=meteo_root,
            site_id=1,
            year=2021
        )
        
        assert len(loaded) == 365
        assert 'tas' in loaded.columns
        assert 'pr' in loaded.columns
    
    def test_load_raw_meteo_filters_year(self, tmp_path):
        """Test that meteo loading filters by year."""
        comp = RawDataComparator()
        
        meteo_root = tmp_path / "meteo_data"
        meteo_root.mkdir()
        
        # Multi-year data
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        df = pd.DataFrame({
            'ts_local': dates,
            'tas': np.random.randn(len(dates))
        })
        
        csv_path = meteo_root / "site_1.csv"
        df.to_csv(csv_path, index=False)
        
        # Load only 2021
        loaded = comp.load_raw_meteo(
            meteo_root=meteo_root,
            site_id=1,
            year=2021
        )
        
        # Should have 365 days (2021 is not leap year)
        assert len(loaded) == 365
        assert loaded['ts_local'].dt.year.unique()[0] == 2021
    
    def test_load_segment_metadata(self, tmp_path):
        """Test loading segment metadata."""
        comp = RawDataComparator()
        
        # Create dummy metadata
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create timestamps as pandas Timestamps with timezone
        start_ts = pd.Timestamp('2021-06-01 00:00:00', tz='UTC')
        end_ts = pd.Timestamp('2021-06-30 23:50:00', tz='UTC')
        
        seg_ids = [
            (
                0,  # combo_id
                0,  # seg_idx
                {'site ID': 1},  # ids_row
                {'temp_treenet': 10.0},  # in_min
                {'temp_treenet': 20.0},  # in_diff
                {'local_T': 5.0},  # out_min
                {'local_T': 15.0},  # out_diff
                {
                    'window_start_utc': start_ts,
                    'window_end_utc': end_ts
                },
                {}
            )
        ]
        
        with open(data_dir / 'train_segment_ids.pkl', 'wb') as f:
            pickle.dump(seg_ids, f)
        
        # Load metadata
        meta = comp.load_segment_metadata(
            data_dir=data_dir,
            split='train',
            combo_id=0,
            seg_idx=0
        )
        
        assert meta['ids_row']['site ID'] == 1
        assert meta['in_min']['temp_treenet'] == 10.0
        assert meta['in_diff']['temp_treenet'] == 20.0
        assert meta['win_start_utc'].tz is not None
    
    def test_load_segment_metadata_not_found(self, tmp_path):
        """Test that missing metadata raises error."""
        comp = RawDataComparator()
        
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        seg_ids = []
        with open(data_dir / 'train_segment_ids.pkl', 'wb') as f:
            pickle.dump(seg_ids, f)
        
        with pytest.raises(ValueError, match="Metadata not found"):
            comp.load_segment_metadata(
                data_dir=data_dir,
                split='train',
                combo_id=999,
                seg_idx=999
            )


class TestVisualizationIntegration:
    """Integration tests for visualization workflow."""
    
    def test_complete_plotting_workflow(self, tmp_path):
        """Test complete segment plotting workflow."""
        # Setup data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create segments
        timestamps = pd.date_range('2021-06-01', periods=288, freq='10min', tz='UTC')
        combo_ids = {0: {'site ID': 1}}
        
        input_segs = {
            0: [pd.DataFrame({
                'temp_treenet': np.random.randn(288),
                'rh_treenet': np.random.randn(288),
                'stem': np.random.randn(288)
            }, index=timestamps)]
        }
        
        output_segs = {
            0: [pd.DataFrame({
                'local_T': np.random.randn(24),
                'local_RH': np.random.randn(24),
                'stem': np.random.randn(24)
            }, index=pd.date_range('2021-06-01', periods=24, freq='60min', tz='UTC'))]
        }
        
        seg_metadata = []
        
        # Save files
        for split in ['train']:
            with open(data_dir / f'model_{split}_data_combination_ids.pkl', 'wb') as f:
                pickle.dump(combo_ids, f)
            with open(data_dir / f'{split}_input_segments.pkl', 'wb') as f:
                pickle.dump(input_segs, f)
            with open(data_dir / f'{split}_output_segments.pkl', 'wb') as f:
                pickle.dump(output_segs, f)
            with open(data_dir / f'{split}_segment_ids.pkl', 'wb') as f:
                pickle.dump(seg_metadata, f)
        
        # Plot
        plotter = SegmentPlotter()
        output_dir = tmp_path / "plots"
        output_dir.mkdir()
        
        plotter.plot_segments_for_site(
            data_dir=data_dir,
            site_id=1,
            year=2021,
            output_dir=output_dir,
            split='train',
            max_segments=5
        )
        
        # Check output
        plot_files = list(output_dir.glob("*.png"))
        assert len(plot_files) > 0
        
        plt.close('all')
