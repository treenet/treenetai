"""
Metrics for evaluating gap filling performance.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


class GapFillingMetrics:
    """
    Calculate metrics for gap filling evaluation.
    
    Metrics include:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² (Coefficient of Determination)
    - Bias
    """
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error for masked regions.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            mask: Binary mask (0 where gap, 1 otherwise)
            
        Returns:
            MAE value
        """
        gap_mask = (mask == 0)
        if not np.any(gap_mask):
            return 0.0
        
        errors = np.abs(y_true[gap_mask] - y_pred[gap_mask])
        return float(np.mean(errors))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error for masked regions.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            mask: Binary mask
            
        Returns:
            RMSE value
        """
        gap_mask = (mask == 0)
        if not np.any(gap_mask):
            return 0.0
        
        squared_errors = (y_true[gap_mask] - y_pred[gap_mask]) ** 2
        return float(np.sqrt(np.mean(squared_errors)))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate R² score for masked regions.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            mask: Binary mask
            
        Returns:
            R² value
        """
        gap_mask = (mask == 0)
        if not np.any(gap_mask):
            return 0.0
        
        y_true_gap = y_true[gap_mask]
        y_pred_gap = y_pred[gap_mask]
        
        ss_res = np.sum((y_true_gap - y_pred_gap) ** 2)
        ss_tot = np.sum((y_true_gap - np.mean(y_true_gap)) ** 2)
        
        if ss_tot < 1e-10:
            return 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    @staticmethod
    def bias(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate bias (mean error) for masked regions.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            mask: Binary mask
            
        Returns:
            Bias value
        """
        gap_mask = (mask == 0)
        if not np.any(gap_mask):
            return 0.0
        
        errors = y_pred[gap_mask] - y_true[gap_mask]
        return float(np.mean(errors))
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            mask: Binary mask
            
        Returns:
            Dictionary of metric names and values
        """
        return {
            'mae': GapFillingMetrics.mae(y_true, y_pred, mask),
            'rmse': GapFillingMetrics.rmse(y_true, y_pred, mask),
            'r2': GapFillingMetrics.r2_score(y_true, y_pred, mask),
            'bias': GapFillingMetrics.bias(y_true, y_pred, mask)
        }
    
    @staticmethod
    def compute_per_channel_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: np.ndarray,
        channel_names: list
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each channel separately.
        
        Args:
            y_true: Ground truth, shape (batch, time, channels)
            y_pred: Predictions, shape (batch, time, channels)
            mask: Binary mask, shape (batch, time, channels)
            channel_names: List of channel names
            
        Returns:
            Dictionary mapping channel name -> metrics dict
        """
        n_channels = y_true.shape[-1]
        results = {}
        
        for i, name in enumerate(channel_names[:n_channels]):
            y_true_ch = y_true[..., i]
            y_pred_ch = y_pred[..., i]
            mask_ch = mask[..., i]
            
            results[name] = GapFillingMetrics.compute_all_metrics(
                y_true_ch, y_pred_ch, mask_ch
            )
        
        return results
    
    @staticmethod
    def baseline_linear_interpolation(
        x: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Simple baseline: linear interpolation across gaps.
        
        Args:
            x: Input array with gaps (masked values are 0)
            mask: Binary mask (0 where gap)
            
        Returns:
            Array with gaps filled by linear interpolation
        """
        result = x.copy()
        
        for ch in range(x.shape[-1]):
            x_ch = x[..., ch]
            mask_ch = mask[..., ch]
            
            # Find gap boundaries
            gap_indices = np.where(mask_ch == 0)[0]
            
            if len(gap_indices) == 0:
                continue
            
            # Simple linear interpolation
            valid_indices = np.where(mask_ch == 1)[0]
            if len(valid_indices) < 2:
                continue
            
            valid_values = x_ch[valid_indices]
            
            # Interpolate
            result[..., ch] = np.interp(
                np.arange(len(x_ch)),
                valid_indices,
                valid_values
            )
        
        return result
