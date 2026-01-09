"""
Gap injection for training data augmentation.

IMPORTANT: Gaps are ONLY injected into the first 3 channels (local sensor data):
- Channel 0: temp_treenet (local air temperature)
- Channel 1: rh_treenet (local relative humidity)  
- Channel 2: stem (dendrometer stem radius change)

Channels 3-10 (global meteo data) are NEVER gapped because:
1. They represent clean, gap-free global meteorological data
2. They serve as auxiliary information to help the model fill gaps
3. Their coarse spatial/temporal resolution makes them reliable reference data
4. Day-of-year (channel 10) provides critical temporal context

The model learns to use correlations between local sensor data and global meteo
data to reconstruct gaps in the local measurements.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np


# Channels that can have gaps injected (local sensor data only)
GAPPABLE_CHANNELS = [0, 1, 2]  # temp_treenet, rh_treenet, stem

# Channel descriptions for reference
CHANNEL_NAMES = {
    0: 'temp_treenet',   # Local air temperature (below canopy)
    1: 'rh_treenet',     # Local relative humidity (below canopy)
    2: 'stem',           # Dendrometer stem radius change
    3: 'tas',            # Global mean air temperature (daily)
    4: 'tasmax',         # Global max air temperature (daily)
    5: 'tasmin',         # Global min air temperature (daily)
    6: 'rh',             # Global relative humidity (daily)
    7: 'vpd',            # Global vapor pressure deficit (daily)
    8: 'gh',             # Global horizontal irradiance (daily)
    9: 'pr',             # Global precipitation (daily)
    10: 'doy',           # Day of year (1-365)
}


class GapInjector:
    """
    Injects synthetic gaps into time series data for training.
    
    CRITICAL: Gaps are ONLY injected into the first 3 channels (local sensor data).
    Global meteo channels (3-10) are never gapped as they provide clean reference
    data that helps the model learn to reconstruct local measurements.
    """
    
    def __init__(
        self,
        min_gap_days: int = 1,
        max_gap_days: int = 12,
        min_gaps_per_segment: int = 1,
        max_gaps_per_segment: int = 3,
        gap_channel_prob: float = 0.5,
        random_seed: Optional[int] = 42,
        gappable_channels: Optional[List[int]] = None
    ):
        """
        Initialize gap injector.
        
        Args:
            min_gap_days: Minimum gap length in days
            max_gap_days: Maximum gap length in days
            min_gaps_per_segment: Minimum number of gaps per segment
            max_gaps_per_segment: Maximum number of gaps per segment
            gap_channel_prob: Probability of gapping each eligible channel
            random_seed: Random seed for reproducibility
            gappable_channels: List of channel indices that can be gapped.
                              Default is [0, 1, 2] (local sensor channels only).
        """
        self.min_gap_days = min_gap_days
        self.max_gap_days = max_gap_days
        self.min_gaps = min_gaps_per_segment
        self.max_gaps = max_gaps_per_segment
        self.gap_channel_prob = gap_channel_prob
        
        # Only allow gaps in local sensor channels by default
        self.gappable_channels = gappable_channels if gappable_channels is not None else GAPPABLE_CHANNELS
        
        self.rng = np.random.default_rng(random_seed)
    
    def inject_gaps(
        self,
        x: np.ndarray,
        steps_per_hour: int = 6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject random gaps into a single segment.
        
        Gaps are ONLY injected into gappable channels (default: first 3 channels).
        Global meteo channels are never modified.
        
        Gaps are created by:
        1. Setting values to 0
        2. Creating a mask (0 where gap, 1 otherwise)
        
        Args:
            x: Input array of shape (timesteps, channels)
            steps_per_hour: Number of timesteps per hour (6 for 10-min data)
            
        Returns:
            Tuple of (x_gapped, mask)
            - x_gapped: Input with gaps (shape: (timesteps, channels))
            - mask: Binary mask (shape: (timesteps, channels))
        
        Example:
            >>> injector = GapInjector()
            >>> x = np.random.randn(4320, 11)  # 30 days, 11 channels
            >>> x_gapped, mask = injector.inject_gaps(x)
            >>> # Gaps only in channels 0, 1, 2
            >>> np.sum(mask[:, 3:] == 0)  # No gaps in meteo channels
            0
        """
        x_gapped = x.copy().astype(np.float32)
        mask = np.ones_like(x, dtype=np.float32)
        
        timesteps, n_channels = x.shape
        
        # Determine number of gaps to inject
        n_gaps = self.rng.integers(self.min_gaps, self.max_gaps + 1)
        
        # Select channels to gap (ONLY from gappable channels)
        channels_to_gap = []
        for ch in self.gappable_channels:
            if ch < n_channels and self.rng.random() < self.gap_channel_prob:
                channels_to_gap.append(ch)
        
        # Ensure at least one gappable channel is selected
        if not channels_to_gap:
            channels_to_gap = [self.rng.choice(self.gappable_channels)]
        
        # Inject gaps
        for _ in range(n_gaps):
            # Random gap length in days
            gap_days = self.rng.integers(self.min_gap_days, self.max_gap_days + 1)
            gap_steps = gap_days * 24 * steps_per_hour
            
            # Ensure gap fits in segment
            if gap_steps >= timesteps:
                gap_steps = timesteps // 2
            
            # Random start position
            max_start = timesteps - gap_steps
            if max_start < 0:
                continue
            
            gap_start = self.rng.integers(0, max_start + 1)
            gap_end = gap_start + gap_steps
            
            # Random channel to gap
            channel = self.rng.choice(channels_to_gap)
            
            # Apply gap
            x_gapped[gap_start:gap_end, channel] = 0.0
            mask[gap_start:gap_end, channel] = 0.0
        
        return x_gapped, mask
    
    def inject_gaps_batch(
        self,
        X: np.ndarray,
        steps_per_hour: int = 6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject gaps into a batch of segments.
        
        Args:
            X: Batch of inputs, shape (batch_size, timesteps, channels)
            steps_per_hour: Number of timesteps per hour
            
        Returns:
            Tuple of (X_gapped, masks)
            - X_gapped: Batch with gaps, shape (batch_size, timesteps, channels)
            - masks: Binary masks, shape (batch_size, timesteps, channels)
        """
        batch_size = X.shape[0]
        X_gapped = np.zeros_like(X)
        masks = np.zeros_like(X)
        
        for i in range(batch_size):
            X_gapped[i], masks[i] = self.inject_gaps(X[i], steps_per_hour)
        
        return X_gapped, masks
    
    def create_gap_mask_only(
        self,
        shape: Tuple[int, ...],
        steps_per_hour: int = 6
    ) -> np.ndarray:
        """
        Create gap mask without modifying data.
        
        Useful for applying gaps at inference time.
        
        Args:
            shape: Shape of data (timesteps, channels) or (batch, timesteps, channels)
            steps_per_hour: Number of timesteps per hour
            
        Returns:
            Binary mask array
        """
        if len(shape) == 2:
            # Single segment
            dummy_x = np.zeros(shape)
            _, mask = self.inject_gaps(dummy_x, steps_per_hour)
            return mask
        
        elif len(shape) == 3:
            # Batch
            batch_size, timesteps, channels = shape
            masks = np.zeros(shape)
            
            for i in range(batch_size):
                dummy_x = np.zeros((timesteps, channels))
                _, masks[i] = self.inject_gaps(dummy_x, steps_per_hour)
            
            return masks
        
        else:
            raise ValueError(f"Invalid shape: {shape}")


class GapGenerator:
    """
    Alternative gap generator with more control over gap patterns.
    
    This allows for specific gap scenarios:
    - Contiguous gaps
    - Random sparse gaps
    - Channel-specific patterns
    """
    
    @staticmethod
    def create_contiguous_gap(
        length: int,
        gap_start: int,
        gap_length: int,
        n_channels: int,
        channel_idx: int
    ) -> np.ndarray:
        """
        Create a mask with a single contiguous gap in one channel.
        
        Args:
            length: Total length of time series
            gap_start: Start index of gap
            gap_length: Length of gap in timesteps
            n_channels: Number of channels
            channel_idx: Which channel to gap
            
        Returns:
            Binary mask array of shape (length, n_channels)
        """
        mask = np.ones((length, n_channels), dtype=np.float32)
        gap_end = min(gap_start + gap_length, length)
        mask[gap_start:gap_end, channel_idx] = 0.0
        return mask
    
    @staticmethod
    def create_multi_channel_gap(
        length: int,
        gap_start: int,
        gap_length: int,
        n_channels: int,
        channel_indices: List[int]
    ) -> np.ndarray:
        """
        Create a mask with gaps in multiple channels at the same time.
        
        Args:
            length: Total length of time series
            gap_start: Start index of gap
            gap_length: Length of gap in timesteps
            n_channels: Number of channels
            channel_indices: List of channel indices to gap
            
        Returns:
            Binary mask array
        """
        mask = np.ones((length, n_channels), dtype=np.float32)
        gap_end = min(gap_start + gap_length, length)
        
        for ch in channel_indices:
            if 0 <= ch < n_channels:
                mask[gap_start:gap_end, ch] = 0.0
        
        return mask
    
    @staticmethod
    def create_random_sparse_gaps(
        length: int,
        n_channels: int,
        gap_prob: float = 0.1,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Create random sparse gaps (individual missing values).
        
        Args:
            length: Total length of time series
            n_channels: Number of channels
            gap_prob: Probability of each value being missing
            rng: Random number generator
            
        Returns:
            Binary mask array
        """
        if rng is None:
            rng = np.random.default_rng()
        
        mask = rng.random((length, n_channels)) > gap_prob
        return mask.astype(np.float32)
    
    @staticmethod
    def create_periodic_gaps(
        length: int,
        n_channels: int,
        period: int,
        gap_length: int,
        channel_idx: int = 0
    ) -> np.ndarray:
        """
        Create periodic gaps with fixed interval.
        
        Useful for simulating sensor maintenance windows.
        
        Args:
            length: Total length of time series
            period: Period between gaps (in timesteps)
            gap_length: Length of each gap
            n_channels: Number of channels
            channel_idx: Which channel to gap
            
        Returns:
            Binary mask array
        """
        mask = np.ones((length, n_channels), dtype=np.float32)
        
        for start in range(0, length, period):
            end = min(start + gap_length, length)
            mask[start:end, channel_idx] = 0.0
        
        return mask
