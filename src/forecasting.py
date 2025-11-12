"""
Traditional and modern forecasting methods for time series analysis.

This module includes ARIMA, Prophet, and other statistical forecasting methods.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging
from dataclasses import dataclass
import warnings

# Traditional forecasting
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

# Darts
try:
    from darts import TimeSeries
    from darts.models import ARIMA as DartsARIMA
    from darts.models import ExponentialSmoothing
    from darts.models import LinearRegressionModel
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    warnings.warn("Darts not available. Install with: pip install darts")

logger = logging.getLogger(__name__)


@dataclass
class ForecastConfig:
    """Configuration for forecasting models."""
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    arima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = False
    forecast_horizon: int = 30
    confidence_level: float = 0.95


class ARIMAForecaster:
    """ARIMA forecasting model."""
    
    def __init__(self, config: ForecastConfig):
        """Initialize ARIMA forecaster.
        
        Args:
            config: Forecasting configuration
        """
        self.config = config
        self.model = None
        self.fitted_model = None
        
    def check_stationarity(self, data: np.ndarray) -> Dict[str, Any]:
        """Check if time series is stationary.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        result = adfuller(data)
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def auto_arima(self, data: np.ndarray, 
                   seasonal: bool = True,
                   max_p: int = 5, max_q: int = 5,
                   max_P: int = 2, max_Q: int = 2) -> Tuple[int, int, int]:
        """Automatically determine ARIMA order using pmdarima.
        
        Args:
            data: Time series data
            seasonal: Whether to include seasonal components
            max_p: Maximum AR order
            max_q: Maximum MA order
            max_P: Maximum seasonal AR order
            max_Q: Maximum seasonal MA order
            
        Returns:
            Tuple of (p, d, q) order
        """
        try:
            model = pm.auto_arima(
                data,
                seasonal=seasonal,
                max_p=max_p, max_q=max_q,
                max_P=max_P, max_Q=max_Q,
                m=12 if seasonal else 1,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            order = model.order
            logger.info(f"Auto ARIMA selected order: {order}")
            return order
            
        except Exception as e:
            logger.warning(f"Auto ARIMA failed: {e}. Using default order.")
            return self.config.arima_order
    
    def fit(self, data: np.ndarray, auto_order: bool = True) -> None:
        """Fit ARIMA model to data.
        
        Args:
            data: Time series data
            auto_order: Whether to automatically determine order
        """
        logger.info("Fitting ARIMA model")
        
        if auto_order:
            order = self.auto_arima(data)
        else:
            order = self.config.arima_order
        
        # Create ARIMA model
        self.model = ARIMA(data, order=order)
        self.fitted_model = self.model.fit()
        
        logger.info(f"ARIMA model fitted with order {order}")
        logger.info(f"AIC: {self.fitted_model.aic:.2f}")
    
    def forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.forecast(
            steps=steps,
            alpha=1 - self.config.confidence_level
        )
        
        if isinstance(forecast_result, tuple):
            forecast, conf_int = forecast_result
            lower_bound = conf_int[:, 0]
            upper_bound = conf_int[:, 1]
        else:
            forecast = forecast_result
            # Simple confidence intervals based on standard error
            std_error = np.std(self.fitted_model.resid)
            lower_bound = forecast - 1.96 * std_error
            upper_bound = forecast + 1.96 * std_error
        
        return forecast, lower_bound, upper_bound
    
    def get_model_summary(self) -> str:
        """Get model summary.
        
        Returns:
            Model summary as string
        """
        if self.fitted_model is None:
            return "Model not fitted"
        
        return str(self.fitted_model.summary())


class ProphetForecaster:
    """Prophet forecasting model."""
    
    def __init__(self, config: ForecastConfig):
        """Initialize Prophet forecaster.
        
        Args:
            config: Forecasting configuration
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
        
        self.config = config
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> None:
        """Fit Prophet model to data.
        
        Args:
            data: Time series data
            dates: Optional datetime index
        """
        logger.info("Fitting Prophet model")
        
        # Create DataFrame for Prophet
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
        
        df = pd.DataFrame({
            'ds': dates,
            'y': data
        })
        
        # Initialize Prophet model
        self.model = Prophet(
            yearly_seasonality=self.config.prophet_yearly_seasonality,
            weekly_seasonality=self.config.prophet_weekly_seasonality,
            daily_seasonality=self.config.prophet_daily_seasonality,
            seasonality_mode='multiplicative'
        )
        
        # Fit model
        self.fitted_model = self.model.fit(df)
        
        logger.info("Prophet model fitted successfully")
    
    def forecast(self, steps: int, dates: Optional[pd.DatetimeIndex] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            dates: Optional future dates
            
        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Create future DataFrame
        if dates is None:
            future = self.model.make_future_dataframe(periods=steps)
        else:
            future = pd.DataFrame({'ds': dates})
        
        # Generate forecast
        forecast_df = self.fitted_model.predict(future)
        
        # Extract forecast values
        forecast = forecast_df['yhat'].values[-steps:]
        lower_bound = forecast_df['yhat_lower'].values[-steps:]
        upper_bound = forecast_df['yhat_upper'].values[-steps:]
        
        return forecast, lower_bound, upper_bound
    
    def plot_components(self) -> None:
        """Plot Prophet model components."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before plotting")
        
        self.fitted_model.plot_components()


class DartsForecaster:
    """Darts-based forecasting models."""
    
    def __init__(self, config: ForecastConfig):
        """Initialize Darts forecaster.
        
        Args:
            config: Forecasting configuration
        """
        if not DARTS_AVAILABLE:
            raise ImportError("Darts is not available. Install with: pip install darts")
        
        self.config = config
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: np.ndarray, model_type: str = 'arima') -> None:
        """Fit Darts model to data.
        
        Args:
            data: Time series data
            model_type: Type of model ('arima', 'exponential_smoothing', 'linear_regression')
        """
        logger.info(f"Fitting Darts {model_type} model")
        
        # Convert to Darts TimeSeries
        ts = TimeSeries.from_values(data)
        
        # Create model
        if model_type == 'arima':
            self.model = DartsARIMA(
                order=self.config.arima_order,
                seasonal_order=self.config.arima_seasonal_order
            )
        elif model_type == 'exponential_smoothing':
            self.model = ExponentialSmoothing()
        elif model_type == 'linear_regression':
            self.model = LinearRegressionModel(lags=12)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model
        self.fitted_model = self.model.fit(ts)
        
        logger.info(f"Darts {model_type} model fitted successfully")
    
    def forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Generate forecast
        forecast_ts = self.fitted_model.predict(n=steps)
        
        # Extract values
        forecast = forecast_ts.values().flatten()
        
        # Simple confidence intervals (Darts doesn't always provide them)
        std_error = np.std(forecast) * 0.1  # Rough estimate
        lower_bound = forecast - 1.96 * std_error
        upper_bound = forecast + 1.96 * std_error
        
        return forecast, lower_bound, upper_bound


class EnsembleForecaster:
    """Ensemble forecasting combining multiple models."""
    
    def __init__(self, config: ForecastConfig):
        """Initialize ensemble forecaster.
        
        Args:
            config: Forecasting configuration
        """
        self.config = config
        self.models = {}
        self.weights = {}
        
    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Forecasting model
            weight: Model weight
        """
        self.models[name] = model
        self.weights[name] = weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
    
    def fit(self, data: np.ndarray) -> None:
        """Fit all models in the ensemble.
        
        Args:
            data: Time series data
        """
        logger.info("Fitting ensemble models")
        
        for name, model in self.models.items():
            logger.info(f"Fitting {name}")
            model.fit(data)
    
    def forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ensemble forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        forecasts = {}
        lower_bounds = {}
        upper_bounds = {}
        
        # Get forecasts from all models
        for name, model in self.models.items():
            try:
                forecast, lower, upper = model.forecast(steps)
                forecasts[name] = forecast
                lower_bounds[name] = lower
                upper_bounds[name] = upper
            except Exception as e:
                logger.warning(f"Model {name} failed to forecast: {e}")
        
        if not forecasts:
            raise ValueError("No models successfully generated forecasts")
        
        # Weighted average of forecasts
        ensemble_forecast = np.zeros(steps)
        ensemble_lower = np.zeros(steps)
        ensemble_upper = np.zeros(steps)
        
        for name, forecast in forecasts.items():
            weight = self.weights[name]
            ensemble_forecast += weight * forecast
            ensemble_lower += weight * lower_bounds[name]
            ensemble_upper += weight * upper_bounds[name]
        
        return ensemble_forecast, ensemble_lower, ensemble_upper


def seasonal_decomposition(data: np.ndarray, 
                         model: str = 'additive',
                         period: int = 12) -> Dict[str, np.ndarray]:
    """Perform seasonal decomposition.
    
    Args:
        data: Time series data
        model: Decomposition model ('additive' or 'multiplicative')
        period: Seasonal period
        
    Returns:
        Dictionary with trend, seasonal, and residual components
    """
    # Convert to pandas Series for statsmodels
    ts = pd.Series(data)
    
    decomposition = seasonal_decompose(
        ts, 
        model=model, 
        period=period,
        extrapolate_trend='freq'
    )
    
    return {
        'trend': decomposition.trend.values,
        'seasonal': decomposition.seasonal.values,
        'residual': decomposition.resid.values,
        'observed': decomposition.observed.values
    }


if __name__ == "__main__":
    # Example usage
    config = ForecastConfig()
    
    # Generate sample data
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    data = np.sin(t) + 0.1 * np.random.randn(100)
    
    # Test ARIMA
    arima_forecaster = ARIMAForecaster(config)
    arima_forecaster.fit(data)
    forecast, lower, upper = arima_forecaster.forecast(10)
    
    print(f"ARIMA forecast: {forecast[:5]}")
    print(f"ARIMA AIC: {arima_forecaster.fitted_model.aic:.2f}")
