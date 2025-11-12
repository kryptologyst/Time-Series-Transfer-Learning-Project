"""
Comprehensive unit tests for the Time Series Transfer Learning project.

This module contains tests for all major components including
data generation, models, forecasting, anomaly detection, and visualization.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import our modules
from src.data_generation import (
    TimeSeriesConfig, TimeSeriesGenerator, 
    create_sequences, split_time_series,
    generate_transfer_learning_datasets
)
from src.models import (
    ModelConfig, LSTMModel, GRUModel, TransformerModel,
    TransferLearningTrainer, create_model
)
from src.forecasting import (
    ForecastConfig, ARIMAForecaster, ProphetForecaster,
    seasonal_decomposition
)
from src.anomaly_detection import (
    AnomalyConfig, StatisticalAnomalyDetector, MLAnomalyDetector,
    AutoencoderAnomalyDetector
)
from src.visualization import (
    PlotConfig, TimeSeriesVisualizer, InteractiveVisualizer
)


class TestDataGeneration(unittest.TestCase):
    """Test cases for data generation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TimeSeriesConfig(
            length=100,
            noise_level=0.1,
            phase_shift=np.pi/4,
            random_seed=42
        )
        self.generator = TimeSeriesGenerator(self.config)
    
    def test_time_series_config(self):
        """Test TimeSeriesConfig dataclass."""
        config = TimeSeriesConfig()
        self.assertEqual(config.length, 500)
        self.assertEqual(config.noise_level, 0.1)
        self.assertEqual(config.phase_shift, 0.0)
        self.assertIsNone(config.random_seed)
    
    def test_sine_wave_generation(self):
        """Test sine wave generation."""
        series = self.generator.generate_sine_wave()
        
        self.assertEqual(len(series), self.config.length)
        self.assertIsInstance(series, np.ndarray)
        self.assertTrue(np.all(np.isfinite(series)))
    
    def test_arima_process_generation(self):
        """Test ARIMA process generation."""
        series = self.generator.generate_arima_process()
        
        self.assertEqual(len(series), self.config.length)
        self.assertIsInstance(series, np.ndarray)
        self.assertTrue(np.all(np.isfinite(series)))
    
    def test_trending_series_generation(self):
        """Test trending series generation."""
        series = self.generator.generate_trending_series("linear")
        
        self.assertEqual(len(series), self.config.length)
        self.assertIsInstance(series, np.ndarray)
        self.assertTrue(np.all(np.isfinite(series)))
    
    def test_anomaly_addition(self):
        """Test anomaly addition to time series."""
        series = self.generator.generate_sine_wave()
        series_with_anomalies, anomaly_mask = self.generator.add_anomalies(series)
        
        self.assertEqual(len(series_with_anomalies), len(series))
        self.assertEqual(len(anomaly_mask), len(series))
        self.assertIsInstance(anomaly_mask, np.ndarray)
        self.assertEqual(anomaly_mask.dtype, bool)
    
    def test_create_sequences(self):
        """Test sequence creation for time series."""
        data = np.random.randn(100)
        X, y = create_sequences(data, sequence_length=10)
        
        self.assertEqual(X.shape[0], len(data) - 10)
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(len(y), len(data) - 10)
    
    def test_split_time_series(self):
        """Test time series splitting."""
        data = np.random.randn(100)
        train, val, test = split_time_series(data, 0.6, 0.2, 0.2)
        
        self.assertEqual(len(train), 60)
        self.assertEqual(len(val), 20)
        self.assertEqual(len(test), 20)
        self.assertEqual(len(train) + len(val) + len(test), len(data))
    
    def test_generate_transfer_learning_datasets(self):
        """Test transfer learning dataset generation."""
        source_config = TimeSeriesConfig(length=100, random_seed=42)
        target_config = TimeSeriesConfig(length=100, phase_shift=np.pi/4, random_seed=42)
        
        datasets = generate_transfer_learning_datasets(source_config, target_config)
        
        self.assertIn('source', datasets)
        self.assertIn('target', datasets)
        self.assertIn('data', datasets['source'])
        self.assertIn('X', datasets['source'])
        self.assertIn('y', datasets['source'])
        self.assertIn('train', datasets['target'])
        self.assertIn('val', datasets['target'])
        self.assertIn('test', datasets['target'])


class TestModels(unittest.TestCase):
    """Test cases for models module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            hidden_dim=16,
            num_layers=1,
            dropout=0.1,
            learning_rate=0.01,
            batch_size=16,
            epochs_pretrain=2,
            epochs_finetune=1,
            sequence_length=10
        )
    
    def test_model_config(self):
        """Test ModelConfig dataclass."""
        config = ModelConfig()
        self.assertEqual(config.hidden_dim, 32)
        self.assertEqual(config.num_layers, 1)
        self.assertEqual(config.dropout, 0.1)
    
    def test_lstm_model(self):
        """Test LSTM model."""
        model = LSTMModel(input_size=1, hidden_dim=16, num_layers=1)
        
        # Test forward pass
        x = torch.randn(4, 10, 1)  # batch_size=4, seq_len=10, input_size=1
        output = model(x)
        
        self.assertEqual(output.shape, (4,))  # batch_size=4, output_size=1
    
    def test_gru_model(self):
        """Test GRU model."""
        model = GRUModel(input_size=1, hidden_dim=16, num_layers=1)
        
        # Test forward pass
        x = torch.randn(4, 10, 1)
        output = model(x)
        
        self.assertEqual(output.shape, (4,))
    
    def test_transformer_model(self):
        """Test Transformer model."""
        model = TransformerModel(input_size=1, d_model=16, nhead=2, num_layers=1)
        
        # Test forward pass
        x = torch.randn(4, 10, 1)
        output = model(x)
        
        self.assertEqual(output.shape, (4,))
    
    def test_transfer_learning_trainer(self):
        """Test TransferLearningTrainer."""
        model = LSTMModel(input_size=1, hidden_dim=16)
        trainer = TransferLearningTrainer(model, self.config)
        
        # Create dummy data
        X_source = np.random.randn(50, 10)
        y_source = np.random.randn(50)
        X_target = np.random.randn(30, 10)
        y_target = np.random.randn(30)
        
        # Test pretraining
        trainer.pretrain(X_source, y_source)
        self.assertEqual(len(trainer.history['pretrain_loss']), self.config.epochs_pretrain)
        
        # Test fine-tuning
        trainer.finetune(X_target, y_target)
        self.assertEqual(len(trainer.history['finetune_loss']), self.config.epochs_finetune)
        
        # Test prediction
        predictions = trainer.predict(X_target)
        self.assertEqual(len(predictions), len(X_target))
    
    def test_create_model_factory(self):
        """Test model factory function."""
        lstm_model = create_model('lstm', self.config)
        gru_model = create_model('gru', self.config)
        transformer_model = create_model('transformer', self.config)
        
        self.assertIsInstance(lstm_model, LSTMModel)
        self.assertIsInstance(gru_model, GRUModel)
        self.assertIsInstance(transformer_model, TransformerModel)
        
        with self.assertRaises(ValueError):
            create_model('unknown', self.config)


class TestForecasting(unittest.TestCase):
    """Test cases for forecasting module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ForecastConfig(
            arima_order=(1, 1, 1),
            forecast_horizon=10
        )
        self.data = np.random.randn(100)
    
    def test_forecast_config(self):
        """Test ForecastConfig dataclass."""
        config = ForecastConfig()
        self.assertEqual(config.arima_order, (1, 1, 1))
        self.assertEqual(config.forecast_horizon, 30)
    
    def test_arima_forecaster(self):
        """Test ARIMA forecaster."""
        forecaster = ARIMAForecaster(self.config)
        
        # Test stationarity check
        stationarity_result = forecaster.check_stationarity(self.data)
        self.assertIn('adf_statistic', stationarity_result)
        self.assertIn('p_value', stationarity_result)
        self.assertIn('is_stationary', stationarity_result)
        
        # Test fitting
        forecaster.fit(self.data, auto_order=False)
        self.assertIsNotNone(forecaster.fitted_model)
        
        # Test forecasting
        forecast, lower, upper = forecaster.forecast(10)
        self.assertEqual(len(forecast), 10)
        self.assertEqual(len(lower), 10)
        self.assertEqual(len(upper), 10)
    
    def test_auto_arima(self):
        """Test automatic ARIMA order selection."""
        forecaster = ARIMAForecaster(self.config)
        
        # Test with simple data
        simple_data = np.random.randn(50)
        order = forecaster.auto_arima(simple_data, seasonal=False)
        
        self.assertIsInstance(order, tuple)
        self.assertEqual(len(order), 3)
    
    def test_seasonal_decomposition(self):
        """Test seasonal decomposition."""
        # Create data with seasonality
        t = np.linspace(0, 4*np.pi, 100)
        seasonal_data = np.sin(t) + 0.1 * np.random.randn(100)
        
        decomposition = seasonal_decomposition(seasonal_data, period=25)
        
        self.assertIn('trend', decomposition)
        self.assertIn('seasonal', decomposition)
        self.assertIn('residual', decomposition)
        self.assertIn('observed', decomposition)
        
        for component in decomposition.values():
            self.assertEqual(len(component), len(seasonal_data))


class TestAnomalyDetection(unittest.TestCase):
    """Test cases for anomaly detection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AnomalyConfig(
            contamination=0.1,
            z_threshold=3.0,
            random_seed=42
        )
        self.data = np.random.randn(100)
    
    def test_anomaly_config(self):
        """Test AnomalyConfig dataclass."""
        config = AnomalyConfig()
        self.assertEqual(config.contamination, 0.1)
        self.assertEqual(config.z_threshold, 3.0)
    
    def test_statistical_anomaly_detector(self):
        """Test statistical anomaly detector."""
        detector = StatisticalAnomalyDetector(self.config)
        
        # Test Z-score detection
        anomaly_mask, thresholds = detector.z_score_detection(self.data)
        self.assertEqual(len(anomaly_mask), len(self.data))
        self.assertIsInstance(anomaly_mask, np.ndarray)
        self.assertEqual(anomaly_mask.dtype, bool)
        
        # Test IQR detection
        anomaly_mask, thresholds = detector.iqr_detection(self.data)
        self.assertEqual(len(anomaly_mask), len(self.data))
        
        # Test modified Z-score detection
        anomaly_mask, thresholds = detector.modified_z_score_detection(self.data)
        self.assertEqual(len(anomaly_mask), len(self.data))
    
    def test_ml_anomaly_detector(self):
        """Test machine learning anomaly detector."""
        detector = MLAnomalyDetector(self.config)
        
        # Test Isolation Forest
        anomaly_mask, info = detector.isolation_forest_detection(self.data)
        self.assertEqual(len(anomaly_mask), len(self.data))
        self.assertIn('n_anomalies', info)
        
        # Test One-Class SVM
        anomaly_mask, info = detector.one_class_svm_detection(self.data)
        self.assertEqual(len(anomaly_mask), len(self.data))
        
        # Test DBSCAN
        anomaly_mask, info = detector.dbscan_detection(self.data)
        self.assertEqual(len(anomaly_mask), len(self.data))
    
    def test_autoencoder_anomaly_detector(self):
        """Test autoencoder anomaly detector."""
        detector = AutoencoderAnomalyDetector(self.config)
        
        # Test fitting
        detector.fit(self.data, window_size=10)
        self.assertIsNotNone(detector.model)
        
        # Test anomaly detection
        anomaly_mask, info = detector.detect_anomalies(self.data, window_size=10)
        self.assertEqual(len(anomaly_mask), len(self.data))
        self.assertIn('n_anomalies', info)


class TestVisualization(unittest.TestCase):
    """Test cases for visualization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlotConfig(
            figure_size=(8, 6),
            save_plots=False
        )
        self.visualizer = TimeSeriesVisualizer(self.config)
        self.data = np.random.randn(100)
    
    def test_plot_config(self):
        """Test PlotConfig dataclass."""
        config = PlotConfig()
        self.assertEqual(config.figure_size, (12, 8))
        self.assertEqual(config.style, "seaborn-v0_8")
    
    def test_time_series_visualizer(self):
        """Test TimeSeriesVisualizer."""
        # Test basic plotting (without showing)
        with patch('matplotlib.pyplot.show'):
            self.visualizer.plot_time_series(self.data, title="Test Plot")
    
    def test_interactive_visualizer(self):
        """Test InteractiveVisualizer."""
        visualizer = InteractiveVisualizer(self.config)
        
        # Test interactive forecast creation
        forecast = np.random.randn(20)
        fig = visualizer.create_interactive_forecast(self.data, forecast)
        
        self.assertIsInstance(fig, go.Figure)
    
    def test_anomaly_plotting(self):
        """Test anomaly plotting."""
        anomaly_mask = np.random.choice([True, False], size=len(self.data), p=[0.1, 0.9])
        
        with patch('matplotlib.pyplot.show'):
            self.visualizer.plot_anomalies(self.data, anomaly_mask, title="Test Anomalies")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config
        self.config = {
            'data': {
                'source': {'length': 50, 'noise': 0.1, 'shift': 0.0},
                'target': {'length': 50, 'noise': 0.1, 'shift': 0.785},
                'sequence_length': 10
            },
            'models': {
                'lstm': {
                    'hidden_dim': 16,
                    'learning_rate': 0.01,
                    'epochs_pretrain': 2,
                    'epochs_finetune': 1
                }
            },
            'logging': {'level': 'WARNING'},
            'paths': {
                'data_dir': self.temp_dir,
                'models_dir': self.temp_dir,
                'plots_dir': self.temp_dir,
                'logs_dir': self.temp_dir
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        source_config = TimeSeriesConfig(length=50, random_seed=42)
        target_config = TimeSeriesConfig(length=50, phase_shift=np.pi/4, random_seed=42)
        datasets = generate_transfer_learning_datasets(source_config, target_config)
        
        # Train model
        model_config = ModelConfig(hidden_dim=16, epochs_pretrain=2, epochs_finetune=1)
        model = create_model('lstm', model_config)
        trainer = TransferLearningTrainer(model, model_config)
        
        trainer.pretrain(datasets['source']['X'], datasets['source']['y'])
        trainer.finetune(datasets['target']['train']['X'], datasets['target']['train']['y'])
        
        # Run forecasting
        forecast_config = ForecastConfig(forecast_horizon=10)
        arima_forecaster = ARIMAForecaster(forecast_config)
        arima_forecaster.fit(datasets['target']['data'], auto_order=False)
        forecast, lower, upper = arima_forecaster.forecast(10)
        
        # Run anomaly detection
        anomaly_config = AnomalyConfig(contamination=0.1)
        detector = StatisticalAnomalyDetector(anomaly_config)
        anomaly_mask, info = detector.z_score_detection(datasets['target']['data'])
        
        # Verify results
        self.assertIsNotNone(datasets)
        self.assertIsNotNone(trainer)
        self.assertEqual(len(forecast), 10)
        self.assertEqual(len(anomaly_mask), len(datasets['target']['data']))


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling."""
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        config = ModelConfig()
        
        with self.assertRaises(ValueError):
            create_model('invalid_model', config)
    
    def test_invalid_trend_type(self):
        """Test error handling for invalid trend type."""
        config = TimeSeriesConfig()
        generator = TimeSeriesGenerator(config)
        
        with self.assertRaises(ValueError):
            generator.generate_trending_series("invalid_trend")
    
    def test_invalid_split_ratios(self):
        """Test error handling for invalid split ratios."""
        data = np.random.randn(100)
        
        with self.assertRaises(ValueError):
            split_time_series(data, 0.5, 0.3, 0.3)  # Sum > 1.0
    
    def test_model_not_fitted(self):
        """Test error handling when model is not fitted."""
        forecaster = ARIMAForecaster(ForecastConfig())
        
        with self.assertRaises(ValueError):
            forecaster.forecast(10)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataGeneration,
        TestModels,
        TestForecasting,
        TestAnomalyDetection,
        TestVisualization,
        TestIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
