"""
Main application for Time Series Transfer Learning Project.

This module provides a comprehensive interface for time series analysis,
including data generation, model training, forecasting, and anomaly detection.
"""

import numpy as np
import pandas as pd
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

# Import our modules
from src.data_generation import (
    TimeSeriesConfig, TimeSeriesGenerator, 
    generate_transfer_learning_datasets
)
from src.models import (
    ModelConfig, TransferLearningTrainer, 
    create_model
)
from src.forecasting import (
    ForecastConfig, ARIMAForecaster, ProphetForecaster,
    EnsembleForecaster, seasonal_decomposition
)
from src.anomaly_detection import (
    AnomalyConfig, StatisticalAnomalyDetector, MLAnomalyDetector,
    AutoencoderAnomalyDetector, EnsembleAnomalyDetector
)
from src.visualization import (
    PlotConfig, TimeSeriesVisualizer, InteractiveVisualizer,
    create_summary_dashboard
)

# Suppress warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalysisApp:
    """Main application class for time series analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        self._setup_directories()
        
        # Initialize components
        self.data_config = self._create_data_config()
        self.model_config = self._create_model_config()
        self.forecast_config = self._create_forecast_config()
        self.anomaly_config = self._create_anomaly_config()
        self.plot_config = self._create_plot_config()
        
        # Initialize generators and models
        self.data_generator = TimeSeriesGenerator(self.data_config)
        self.visualizer = TimeSeriesVisualizer(self.plot_config)
        self.interactive_visualizer = InteractiveVisualizer(self.plot_config)
        
        logger.info("Time Series Analysis App initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return {}
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        format_str = log_config.get('format', 
                                  '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.FileHandler(log_config.get('file', 'logs/timeseries.log')),
                logging.StreamHandler()
            ]
        )
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        paths = self.config.get('paths', {})
        for path_name, path_value in paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def _create_data_config(self) -> TimeSeriesConfig:
        """Create data generation configuration."""
        data_config = self.config.get('data', {})
        source_config = data_config.get('source', {})
        target_config = data_config.get('target', {})
        
        return TimeSeriesConfig(
            length=source_config.get('length', 500),
            noise_level=source_config.get('noise', 0.1),
            phase_shift=source_config.get('shift', 0.0),
            random_seed=42
        )
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration."""
        model_config = self.config.get('models', {}).get('lstm', {})
        
        return ModelConfig(
            hidden_dim=model_config.get('hidden_dim', 32),
            num_layers=model_config.get('num_layers', 1),
            dropout=model_config.get('dropout', 0.1),
            learning_rate=model_config.get('learning_rate', 0.005),
            batch_size=model_config.get('batch_size', 32),
            epochs_pretrain=model_config.get('epochs_pretrain', 10),
            epochs_finetune=model_config.get('epochs_finetune', 5),
            sequence_length=self.config.get('data', {}).get('sequence_length', 30)
        )
    
    def _create_forecast_config(self) -> ForecastConfig:
        """Create forecasting configuration."""
        forecast_config = self.config.get('models', {}).get('arima', {})
        prophet_config = self.config.get('models', {}).get('prophet', {})
        
        return ForecastConfig(
            arima_order=tuple(forecast_config.get('order', [1, 1, 1])),
            arima_seasonal_order=tuple(forecast_config.get('seasonal_order', [1, 1, 1, 12])),
            prophet_yearly_seasonality=prophet_config.get('yearly_seasonality', True),
            prophet_weekly_seasonality=prophet_config.get('weekly_seasonality', True),
            prophet_daily_seasonality=prophet_config.get('daily_seasonality', False),
            forecast_horizon=self.config.get('data', {}).get('sequence_length', 30)
        )
    
    def _create_anomaly_config(self) -> AnomalyConfig:
        """Create anomaly detection configuration."""
        anomaly_config = self.config.get('anomaly_detection', {})
        
        return AnomalyConfig(
            contamination=anomaly_config.get('isolation_forest', {}).get('contamination', 0.1),
            z_threshold=3.0,
            iqr_multiplier=1.5,
            window_size=10,
            autoencoder_encoding_dim=anomaly_config.get('autoencoder', {}).get('encoding_dim', 16),
            autoencoder_learning_rate=anomaly_config.get('autoencoder', {}).get('learning_rate', 0.001),
            autoencoder_epochs=anomaly_config.get('autoencoder', {}).get('epochs', 50),
            random_state=anomaly_config.get('isolation_forest', {}).get('random_state', 42)
        )
    
    def _create_plot_config(self) -> PlotConfig:
        """Create plotting configuration."""
        viz_config = self.config.get('visualization', {})
        
        return PlotConfig(
            figure_size=tuple(viz_config.get('figure_size', [12, 8])),
            style=viz_config.get('style', 'seaborn-v0_8'),
            color_palette=viz_config.get('color_palette', 'husl'),
            save_plots=viz_config.get('save_plots', True),
            plot_format=viz_config.get('plot_format', 'png'),
            dpi=viz_config.get('dpi', 300)
        )
    
    def generate_data(self) -> Dict[str, Any]:
        """Generate synthetic time series data for transfer learning.
        
        Returns:
            Dictionary containing source and target datasets
        """
        logger.info("Generating synthetic time series data")
        
        # Create source and target configurations
        source_config = TimeSeriesConfig(
            length=self.config.get('data', {}).get('source', {}).get('length', 500),
            noise_level=self.config.get('data', {}).get('source', {}).get('noise', 0.1),
            phase_shift=self.config.get('data', {}).get('source', {}).get('shift', 0.0),
            random_seed=42
        )
        
        target_config = TimeSeriesConfig(
            length=self.config.get('data', {}).get('target', {}).get('length', 500),
            noise_level=self.config.get('data', {}).get('target', {}).get('noise', 0.1),
            phase_shift=self.config.get('data', {}).get('target', {}).get('shift', 0.785),
            random_seed=42
        )
        
        # Generate datasets
        datasets = generate_transfer_learning_datasets(source_config, target_config)
        
        # Save data
        np.save('data/source_data.npy', datasets['source']['data'])
        np.save('data/target_data.npy', datasets['target']['data'])
        
        logger.info("Data generation completed")
        return datasets
    
    def train_transfer_learning_model(self, datasets: Dict[str, Any], 
                                    model_type: str = 'lstm') -> TransferLearningTrainer:
        """Train transfer learning model.
        
        Args:
            datasets: Generated datasets
            model_type: Type of model to train
            
        Returns:
            Trained model trainer
        """
        logger.info(f"Training {model_type.upper()} transfer learning model")
        
        # Create model
        model = create_model(model_type, self.model_config)
        trainer = TransferLearningTrainer(model, self.model_config)
        
        # Pretrain on source data
        trainer.pretrain(datasets['source']['X'], datasets['source']['y'])
        
        # Fine-tune on target data
        trainer.finetune(
            datasets['target']['train']['X'], 
            datasets['target']['train']['y'],
            datasets['target']['val']['X'],
            datasets['target']['val']['y']
        )
        
        # Save model
        trainer.save_model(f'models/{model_type}_transfer_model.pth')
        
        logger.info("Transfer learning model training completed")
        return trainer
    
    def run_forecasting_analysis(self, data: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run comprehensive forecasting analysis.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary of forecasts from different models
        """
        logger.info("Running forecasting analysis")
        
        forecasts = {}
        
        # ARIMA forecasting
        try:
            arima_forecaster = ARIMAForecaster(self.forecast_config)
            arima_forecaster.fit(data, auto_order=True)
            forecast, lower, upper = arima_forecaster.forecast(self.forecast_config.forecast_horizon)
            forecasts['ARIMA'] = (forecast, lower, upper)
            logger.info("ARIMA forecasting completed")
        except Exception as e:
            logger.warning(f"ARIMA forecasting failed: {e}")
        
        # Prophet forecasting (if available)
        try:
            prophet_forecaster = ProphetForecaster(self.forecast_config)
            prophet_forecaster.fit(data)
            forecast, lower, upper = prophet_forecaster.forecast(self.forecast_config.forecast_horizon)
            forecasts['Prophet'] = (forecast, lower, upper)
            logger.info("Prophet forecasting completed")
        except Exception as e:
            logger.warning(f"Prophet forecasting failed: {e}")
        
        # Ensemble forecasting
        try:
            ensemble_forecaster = EnsembleForecaster(self.forecast_config)
            
            # Add ARIMA if available
            if 'ARIMA' in forecasts:
                arima_forecaster = ARIMAForecaster(self.forecast_config)
                arima_forecaster.fit(data, auto_order=True)
                ensemble_forecaster.add_model('ARIMA', arima_forecaster, weight=0.5)
            
            # Add Prophet if available
            if 'Prophet' in forecasts:
                prophet_forecaster = ProphetForecaster(self.forecast_config)
                prophet_forecaster.fit(data)
                ensemble_forecaster.add_model('Prophet', prophet_forecaster, weight=0.5)
            
            if ensemble_forecaster.models:
                ensemble_forecaster.fit(data)
                forecast, lower, upper = ensemble_forecaster.forecast(self.forecast_config.forecast_horizon)
                forecasts['Ensemble'] = (forecast, lower, upper)
                logger.info("Ensemble forecasting completed")
        except Exception as e:
            logger.warning(f"Ensemble forecasting failed: {e}")
        
        return forecasts
    
    def run_anomaly_detection(self, data: np.ndarray) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """Run comprehensive anomaly detection analysis.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary of anomaly detection results
        """
        logger.info("Running anomaly detection analysis")
        
        anomaly_results = {}
        
        # Statistical methods
        stat_detector = StatisticalAnomalyDetector(self.anomaly_config)
        
        # Z-score detection
        anomaly_mask, info = stat_detector.z_score_detection(data)
        anomaly_results['Z-Score'] = (anomaly_mask, info)
        
        # IQR detection
        anomaly_mask, info = stat_detector.iqr_detection(data)
        anomaly_results['IQR'] = (anomaly_mask, info)
        
        # Modified Z-score detection
        anomaly_mask, info = stat_detector.modified_z_score_detection(data)
        anomaly_results['Modified Z-Score'] = (anomaly_mask, info)
        
        # Machine learning methods
        ml_detector = MLAnomalyDetector(self.anomaly_config)
        
        # Isolation Forest
        anomaly_mask, info = ml_detector.isolation_forest_detection(data)
        anomaly_results['Isolation Forest'] = (anomaly_mask, info)
        
        # One-Class SVM
        anomaly_mask, info = ml_detector.one_class_svm_detection(data)
        anomaly_results['One-Class SVM'] = (anomaly_mask, info)
        
        # Autoencoder (if PyTorch is available)
        try:
            autoencoder_detector = AutoencoderAnomalyDetector(self.anomaly_config)
            autoencoder_detector.fit(data)
            anomaly_mask, info = autoencoder_detector.detect_anomalies(data)
            anomaly_results['Autoencoder'] = (anomaly_mask, info)
            logger.info("Autoencoder anomaly detection completed")
        except Exception as e:
            logger.warning(f"Autoencoder anomaly detection failed: {e}")
        
        logger.info("Anomaly detection analysis completed")
        return anomaly_results
    
    def create_visualizations(self, data: np.ndarray, 
                            forecasts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                            anomaly_results: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
                            trainer: Optional[TransferLearningTrainer] = None) -> None:
        """Create comprehensive visualizations.
        
        Args:
            data: Original time series data
            forecasts: Forecasting results
            anomaly_results: Anomaly detection results
            trainer: Optional trained transfer learning model
        """
        logger.info("Creating visualizations")
        
        # Basic time series plot
        self.visualizer.plot_time_series(
            data, 
            title="Generated Time Series Data",
            save_path="plots/time_series.png"
        )
        
        # Forecasting plots
        if forecasts:
            for model_name, (forecast, lower, upper) in forecasts.items():
                self.visualizer.plot_forecast(
                    data, forecast, lower, upper,
                    title=f"{model_name} Forecast",
                    save_path=f"plots/{model_name.lower()}_forecast.png"
                )
            
            # Model comparison
            forecast_dict = {name: forecast for name, (forecast, _, _) in forecasts.items()}
            self.visualizer.plot_model_comparison(
                data, forecast_dict,
                title="Model Comparison",
                save_path="plots/model_comparison.png"
            )
        
        # Anomaly detection plots
        if anomaly_results:
            for method_name, (anomaly_mask, info) in anomaly_results.items():
                self.visualizer.plot_anomalies(
                    data, anomaly_mask,
                    title=f"{method_name} Anomaly Detection",
                    save_path=f"plots/{method_name.lower().replace(' ', '_')}_anomalies.png"
                )
        
        # Seasonal decomposition
        try:
            decomposition = seasonal_decomposition(data)
            self.visualizer.plot_seasonal_decomposition(
                data, decomposition,
                title="Seasonal Decomposition",
                save_path="plots/seasonal_decomposition.png"
            )
        except Exception as e:
            logger.warning(f"Seasonal decomposition failed: {e}")
        
        # Transfer learning results
        if trainer is not None:
            # Plot training history
            plt.figure(figsize=self.plot_config.figure_size)
            
            plt.subplot(1, 2, 1)
            plt.plot(trainer.history['pretrain_loss'], label='Pretrain Loss')
            plt.title('Pretraining Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(trainer.history['finetune_loss'], label='Finetune Loss')
            if trainer.history['val_loss']:
                plt.plot(trainer.history['val_loss'], label='Validation Loss')
            plt.title('Fine-tuning Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("plots/training_history.png", dpi=self.plot_config.dpi, bbox_inches='tight')
            plt.show()
        
        logger.info("Visualizations completed")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete time series analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting complete time series analysis")
        
        # Generate data
        datasets = self.generate_data()
        
        # Train transfer learning model
        trainer = self.train_transfer_learning_model(datasets)
        
        # Run forecasting analysis
        forecasts = self.run_forecasting_analysis(datasets['target']['data'])
        
        # Run anomaly detection
        anomaly_results = self.run_anomaly_detection(datasets['target']['data'])
        
        # Create visualizations
        self.create_visualizations(
            datasets['target']['data'], 
            forecasts, 
            anomaly_results, 
            trainer
        )
        
        # Compile results
        results = {
            'datasets': datasets,
            'trainer': trainer,
            'forecasts': forecasts,
            'anomaly_results': anomaly_results,
            'config': self.config
        }
        
        logger.info("Complete analysis finished successfully")
        return results


def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description='Time Series Transfer Learning Analysis')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='complete',
                       choices=['complete', 'data', 'train', 'forecast', 'anomaly', 'visualize'],
                       help='Analysis mode to run')
    
    args = parser.parse_args()
    
    # Initialize application
    app = TimeSeriesAnalysisApp(args.config)
    
    if args.mode == 'complete':
        results = app.run_complete_analysis()
        print("Complete analysis finished successfully!")
        print(f"Results saved in: data/, models/, plots/")
        
    elif args.mode == 'data':
        datasets = app.generate_data()
        print("Data generation completed!")
        
    elif args.mode == 'train':
        datasets = app.generate_data()
        trainer = app.train_transfer_learning_model(datasets)
        print("Model training completed!")
        
    elif args.mode == 'forecast':
        datasets = app.generate_data()
        forecasts = app.run_forecasting_analysis(datasets['target']['data'])
        print("Forecasting analysis completed!")
        
    elif args.mode == 'anomaly':
        datasets = app.generate_data()
        anomaly_results = app.run_anomaly_detection(datasets['target']['data'])
        print("Anomaly detection completed!")
        
    elif args.mode == 'visualize':
        datasets = app.generate_data()
        forecasts = app.run_forecasting_analysis(datasets['target']['data'])
        anomaly_results = app.run_anomaly_detection(datasets['target']['data'])
        app.create_visualizations(datasets['target']['data'], forecasts, anomaly_results)
        print("Visualization completed!")


if __name__ == "__main__":
    main()
