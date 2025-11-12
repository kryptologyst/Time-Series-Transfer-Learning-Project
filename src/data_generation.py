"""
Data generation module for time series transfer learning.

This module provides functions to generate synthetic time series data
with various patterns including trends, seasonality, and noise.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesConfig:
    """Configuration for time series generation."""
    length: int = 500
    noise_level: float = 0.1
    trend: float = 0.0
    seasonality: bool = True
    seasonal_period: int = 12
    amplitude: float = 1.0
    phase_shift: float = 0.0
    random_seed: Optional[int] = None


class TimeSeriesGenerator:
    """Generator for synthetic time series data."""
    
    def __init__(self, config: TimeSeriesConfig):
        """Initialize the generator with configuration.
        
        Args:
            config: TimeSeriesConfig object with generation parameters
        """
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def generate_sine_wave(self) -> np.ndarray:
        """Generate a sine wave time series.
        
        Returns:
            Generated sine wave time series
        """
        t = np.linspace(0, 20, self.config.length)
        base_signal = self.config.amplitude * np.sin(t + self.config.phase_shift)
        
        # Add trend
        trend_component = self.config.trend * t
        
        # Add seasonality
        seasonal_component = 0
        if self.config.seasonality:
            seasonal_component = 0.3 * np.sin(2 * np.pi * t / self.config.seasonal_period)
        
        # Add noise
        noise = self.config.noise_level * np.random.randn(self.config.length)
        
        return base_signal + trend_component + seasonal_component + noise
    
    def generate_arima_process(self, ar_params: Tuple[float, ...] = (0.5,),
                              ma_params: Tuple[float, ...] = (0.3,),
                              sigma: float = 1.0) -> np.ndarray:
        """Generate ARIMA process time series.
        
        Args:
            ar_params: Autoregressive parameters
            ma_params: Moving average parameters
            sigma: Standard deviation of white noise
            
        Returns:
            Generated ARIMA time series
        """
        n = self.config.length
        ar_order = len(ar_params)
        ma_order = len(ma_params)
        
        # Initialize arrays
        y = np.zeros(n)
        errors = sigma * np.random.randn(n)
        
        # Generate ARIMA process
        for t in range(max(ar_order, ma_order), n):
            # AR component
            ar_component = sum(ar_params[i] * y[t - i - 1] for i in range(ar_order))
            
            # MA component
            ma_component = sum(ma_params[i] * errors[t - i - 1] for i in range(ma_order))
            
            y[t] = ar_component + ma_component + errors[t]
        
        return y
    
    def generate_trending_series(self, trend_type: str = "linear") -> np.ndarray:
        """Generate trending time series.
        
        Args:
            trend_type: Type of trend ('linear', 'exponential', 'logarithmic')
            
        Returns:
            Generated trending time series
        """
        t = np.linspace(0, 10, self.config.length)
        
        if trend_type == "linear":
            trend = self.config.trend * t
        elif trend_type == "exponential":
            trend = self.config.trend * np.exp(0.1 * t)
        elif trend_type == "logarithmic":
            trend = self.config.trend * np.log(1 + t)
        else:
            raise ValueError(f"Unknown trend type: {trend_type}")
        
        # Add seasonal component
        seasonal = 0
        if self.config.seasonality:
            seasonal = 0.5 * np.sin(2 * np.pi * t / self.config.seasonal_period)
        
        # Add noise
        noise = self.config.noise_level * np.random.randn(self.config.length)
        
        return trend + seasonal + noise
    
    def add_anomalies(self, series: np.ndarray, 
                     anomaly_rate: float = 0.05,
                     anomaly_magnitude: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Add anomalies to a time series.
        
        Args:
            series: Input time series
            anomaly_rate: Proportion of points to make anomalous
            anomaly_magnitude: Magnitude of anomalies in standard deviations
            
        Returns:
            Tuple of (series_with_anomalies, anomaly_mask)
        """
        n_anomalies = int(len(series) * anomaly_rate)
        anomaly_indices = np.random.choice(len(series), n_anomalies, replace=False)
        
        series_with_anomalies = series.copy()
        anomaly_mask = np.zeros(len(series), dtype=bool)
        
        std_dev = np.std(series)
        for idx in anomaly_indices:
            anomaly_mask[idx] = True
            # Add random anomaly
            anomaly_value = np.random.choice([-1, 1]) * anomaly_magnitude * std_dev
            series_with_anomalies[idx] += anomaly_value
        
        return series_with_anomalies, anomaly_mask


def create_sequences(data: np.ndarray, 
                    sequence_length: int,
                    target_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series forecasting.
    
    Args:
        data: Input time series data
        sequence_length: Length of input sequences
        target_length: Length of target sequences
        
    Returns:
        Tuple of (X, y) arrays for training
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - target_length + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + target_length])
    
    return np.array(X), np.array(y)


def split_time_series(data: np.ndarray, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split time series into train, validation, and test sets.
    
    Args:
        data: Input time series data
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        
    Returns:
        Tuple of (train, val, test) arrays
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def generate_transfer_learning_datasets(config_source: TimeSeriesConfig,
                                      config_target: TimeSeriesConfig) -> Dict[str, Any]:
    """Generate source and target datasets for transfer learning.
    
    Args:
        config_source: Configuration for source dataset
        config_target: Configuration for target dataset
        
    Returns:
        Dictionary containing source and target datasets
    """
    logger.info("Generating source and target datasets for transfer learning")
    
    # Generate source dataset
    source_generator = TimeSeriesGenerator(config_source)
    source_data = source_generator.generate_sine_wave()
    
    # Generate target dataset
    target_generator = TimeSeriesGenerator(config_target)
    target_data = target_generator.generate_sine_wave()
    
    # Create sequences
    sequence_length = 30
    X_source, y_source = create_sequences(source_data, sequence_length)
    X_target, y_target = create_sequences(target_data, sequence_length)
    
    # Split target data
    train_data, val_data, test_data = split_time_series(target_data)
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_val, y_val = create_sequences(val_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    
    return {
        'source': {
            'data': source_data,
            'X': X_source,
            'y': y_source
        },
        'target': {
            'data': target_data,
            'train': {'data': train_data, 'X': X_train, 'y': y_train},
            'val': {'data': val_data, 'X': X_val, 'y': y_val},
            'test': {'data': test_data, 'X': X_test, 'y': y_test}
        }
    }


if __name__ == "__main__":
    # Example usage
    config = TimeSeriesConfig(
        length=500,
        noise_level=0.1,
        phase_shift=np.pi/4,
        random_seed=42
    )
    
    generator = TimeSeriesGenerator(config)
    series = generator.generate_sine_wave()
    
    print(f"Generated time series with {len(series)} points")
    print(f"Mean: {np.mean(series):.3f}, Std: {np.std(series):.3f}")
