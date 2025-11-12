"""
Comprehensive visualization module for time series analysis.

This module provides various plotting functions for time series data,
forecasts, anomalies, and model comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Tuple, Optional, Dict, Any, List, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    figure_size: Tuple[int, int] = (12, 8)
    style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300
    font_size: int = 12
    line_width: float = 2.0


class TimeSeriesVisualizer:
    """Comprehensive time series visualization class."""
    
    def __init__(self, config: PlotConfig):
        """Initialize visualizer.
        
        Args:
            config: Plot configuration
        """
        self.config = config
        self._setup_style()
        
    def _setup_style(self) -> None:
        """Setup matplotlib style."""
        plt.style.use(self.config.style)
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'figure.figsize': self.config.figure_size,
            'savefig.dpi': self.config.dpi,
            'savefig.format': self.config.plot_format
        })
    
    def plot_time_series(self, data: np.ndarray, 
                        title: str = "Time Series",
                        xlabel: str = "Time",
                        ylabel: str = "Value",
                        dates: Optional[pd.DatetimeIndex] = None,
                        save_path: Optional[str] = None) -> None:
        """Plot basic time series.
        
        Args:
            data: Time series data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            dates: Optional datetime index
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.config.figure_size)
        
        if dates is not None:
            plt.plot(dates, data, linewidth=self.config.line_width)
        else:
            plt.plot(data, linewidth=self.config.line_width)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_multiple_series(self, data_dict: Dict[str, np.ndarray],
                           title: str = "Multiple Time Series",
                           xlabel: str = "Time",
                           ylabel: str = "Value",
                           dates: Optional[pd.DatetimeIndex] = None,
                           save_path: Optional[str] = None) -> None:
        """Plot multiple time series on the same plot.
        
        Args:
            data_dict: Dictionary of series names and data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            dates: Optional datetime index
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.config.figure_size)
        
        colors = sns.color_palette(self.config.color_palette, len(data_dict))
        
        for i, (name, data) in enumerate(data_dict.items()):
            if dates is not None:
                plt.plot(dates, data, label=name, color=colors[i], 
                        linewidth=self.config.line_width)
            else:
                plt.plot(data, label=name, color=colors[i], 
                        linewidth=self.config.line_width)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_forecast(self, actual: np.ndarray,
                     forecast: np.ndarray,
                     lower_bound: Optional[np.ndarray] = None,
                     upper_bound: Optional[np.ndarray] = None,
                     title: str = "Time Series Forecast",
                     xlabel: str = "Time",
                     ylabel: str = "Value",
                     forecast_start: Optional[int] = None,
                     save_path: Optional[str] = None) -> None:
        """Plot time series with forecast.
        
        Args:
            actual: Actual time series data
            forecast: Forecast values
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            forecast_start: Index where forecast starts
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.config.figure_size)
        
        # Determine forecast start
        if forecast_start is None:
            forecast_start = len(actual) - len(forecast)
        
        # Plot actual data
        plt.plot(actual, label='Actual', color='blue', linewidth=self.config.line_width)
        
        # Plot forecast
        forecast_x = range(forecast_start, forecast_start + len(forecast))
        plt.plot(forecast_x, forecast, label='Forecast', color='red', 
                linewidth=self.config.line_width, linestyle='--')
        
        # Plot confidence intervals
        if lower_bound is not None and upper_bound is not None:
            plt.fill_between(forecast_x, lower_bound, upper_bound, 
                           alpha=0.3, color='red', label='Confidence Interval')
        
        # Add vertical line to separate actual and forecast
        plt.axvline(x=forecast_start, color='black', linestyle=':', alpha=0.7)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_anomalies(self, data: np.ndarray,
                      anomaly_mask: np.ndarray,
                      title: str = "Anomaly Detection",
                      xlabel: str = "Time",
                      ylabel: str = "Value",
                      save_path: Optional[str] = None) -> None:
        """Plot time series with highlighted anomalies.
        
        Args:
            data: Time series data
            anomaly_mask: Boolean mask indicating anomalies
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.config.figure_size)
        
        # Plot normal data
        normal_indices = ~anomaly_mask
        plt.plot(np.where(normal_indices)[0], data[normal_indices], 
                color='blue', linewidth=self.config.line_width, label='Normal')
        
        # Plot anomalies
        anomaly_indices = np.where(anomaly_mask)[0]
        if len(anomaly_indices) > 0:
            plt.scatter(anomaly_indices, data[anomaly_mask], 
                       color='red', s=50, label='Anomaly', zorder=5)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_seasonal_decomposition(self, data: np.ndarray,
                                   decomposition: Dict[str, np.ndarray],
                                   title: str = "Seasonal Decomposition",
                                   save_path: Optional[str] = None) -> None:
        """Plot seasonal decomposition.
        
        Args:
            data: Original time series data
            decomposition: Dictionary with trend, seasonal, residual components
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(4, 1, figsize=(self.config.figure_size[0], 
                                                self.config.figure_size[1] * 1.5))
        
        # Original data
        axes[0].plot(data, linewidth=self.config.line_width)
        axes[0].set_title('Original')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(decomposition['trend'], linewidth=self.config.line_width)
        axes[1].set_title('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(decomposition['seasonal'], linewidth=self.config.line_width)
        axes[2].set_title('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(decomposition['residual'], linewidth=self.config.line_width)
        axes[3].set_title('Residual')
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, actual: np.ndarray,
                            forecasts: Dict[str, np.ndarray],
                            title: str = "Model Comparison",
                            xlabel: str = "Time",
                            ylabel: str = "Value",
                            save_path: Optional[str] = None) -> None:
        """Plot comparison of multiple forecasting models.
        
        Args:
            actual: Actual time series data
            forecasts: Dictionary of model names and forecasts
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.config.figure_size)
        
        # Plot actual data
        plt.plot(actual, label='Actual', color='black', linewidth=self.config.line_width)
        
        # Plot forecasts
        colors = sns.color_palette(self.config.color_palette, len(forecasts))
        for i, (name, forecast) in enumerate(forecasts.items()):
            forecast_start = len(actual) - len(forecast)
            forecast_x = range(forecast_start, forecast_start + len(forecast))
            plt.plot(forecast_x, forecast, label=name, color=colors[i], 
                    linewidth=self.config.line_width, linestyle='--')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_error_analysis(self, actual: np.ndarray,
                          predictions: np.ndarray,
                          title: str = "Error Analysis",
                          save_path: Optional[str] = None) -> None:
        """Plot error analysis including residuals and Q-Q plot.
        
        Args:
            actual: Actual values
            predictions: Predicted values
            title: Plot title
            save_path: Optional path to save plot
        """
        errors = actual - predictions
        
        fig, axes = plt.subplots(2, 2, figsize=(self.config.figure_size[0] * 1.2, 
                                               self.config.figure_size[1] * 1.2))
        
        # Residuals plot
        axes[0, 0].scatter(predictions, errors, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time series of residuals
        axes[1, 1].plot(errors, linewidth=self.config.line_width)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()


class InteractiveVisualizer:
    """Interactive visualization using Plotly."""
    
    def __init__(self, config: PlotConfig):
        """Initialize interactive visualizer.
        
        Args:
            config: Plot configuration
        """
        self.config = config
    
    def create_interactive_forecast(self, actual: np.ndarray,
                                  forecast: np.ndarray,
                                  lower_bound: Optional[np.ndarray] = None,
                                  upper_bound: Optional[np.ndarray] = None,
                                  title: str = "Interactive Time Series Forecast") -> go.Figure:
        """Create interactive forecast plot.
        
        Args:
            actual: Actual time series data
            forecast: Forecast values
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=self.config.line_width)
        ))
        
        # Add forecast
        forecast_start = len(actual) - len(forecast)
        forecast_x = list(range(forecast_start, forecast_start + len(forecast)))
        
        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=self.config.line_width, dash='dash')
        ))
        
        # Add confidence intervals
        if lower_bound is not None and upper_bound is not None:
            fig.add_trace(go.Scatter(
                x=forecast_x + forecast_x[::-1],
                y=list(upper_bound) + list(lower_bound[::-1]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_interactive_anomalies(self, data: np.ndarray,
                                   anomaly_mask: np.ndarray,
                                   title: str = "Interactive Anomaly Detection") -> go.Figure:
        """Create interactive anomaly detection plot.
        
        Args:
            data: Time series data
            anomaly_mask: Boolean mask indicating anomalies
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add normal data
        normal_indices = ~anomaly_mask
        fig.add_trace(go.Scatter(
            x=np.where(normal_indices)[0],
            y=data[normal_indices],
            mode='lines',
            name='Normal',
            line=dict(color='blue', width=self.config.line_width)
        ))
        
        # Add anomalies
        anomaly_indices = np.where(anomaly_mask)[0]
        if len(anomaly_indices) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_indices,
                y=data[anomaly_mask],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_model_comparison_dashboard(self, actual: np.ndarray,
                                        forecasts: Dict[str, np.ndarray],
                                        title: str = "Model Comparison Dashboard") -> go.Figure:
        """Create interactive model comparison dashboard.
        
        Args:
            actual: Actual time series data
            forecasts: Dictionary of model names and forecasts
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=self.config.line_width)
        ))
        
        # Add forecasts
        colors = px.colors.qualitative.Set1
        for i, (name, forecast) in enumerate(forecasts.items()):
            forecast_start = len(actual) - len(forecast)
            forecast_x = list(range(forecast_start, forecast_start + len(forecast)))
            
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], 
                         width=self.config.line_width, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig


def create_summary_dashboard(data: np.ndarray,
                           forecasts: Dict[str, np.ndarray],
                           anomalies: Optional[np.ndarray] = None,
                           decomposition: Optional[Dict[str, np.ndarray]] = None) -> go.Figure:
    """Create comprehensive summary dashboard.
    
    Args:
        data: Original time series data
        forecasts: Dictionary of model forecasts
        anomalies: Optional anomaly mask
        decomposition: Optional seasonal decomposition
        
    Returns:
        Plotly figure with subplots
    """
    # Determine number of subplots
    n_subplots = 2  # Main plot + error analysis
    if anomalies is not None:
        n_subplots += 1
    if decomposition is not None:
        n_subplots += 3  # Trend, seasonal, residual
    
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        subplot_titles=['Time Series & Forecasts', 'Error Analysis'] + 
                      (['Anomaly Detection'] if anomalies is not None else []) +
                      (['Trend', 'Seasonal', 'Residual'] if decomposition is not None else []),
        vertical_spacing=0.05
    )
    
    # Main plot
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Actual', 
                            line=dict(color='black')), row=1, col=1)
    
    colors = px.colors.qualitative.Set1
    for i, (name, forecast) in enumerate(forecasts.items()):
        forecast_start = len(data) - len(forecast)
        forecast_x = list(range(forecast_start, forecast_start + len(forecast)))
        
        fig.add_trace(go.Scatter(x=forecast_x, y=forecast, mode='lines', 
                                name=name, line=dict(color=colors[i % len(colors)])),
                     row=1, col=1)
    
    # Error analysis
    if forecasts:
        first_forecast = list(forecasts.values())[0]
        errors = data[-len(first_forecast):] - first_forecast
        fig.add_trace(go.Scatter(y=errors, mode='lines', name='Residuals',
                                line=dict(color='red')), row=2, col=1)
    
    # Anomaly detection
    if anomalies is not None:
        anomaly_indices = np.where(anomalies)[0]
        fig.add_trace(go.Scatter(x=anomaly_indices, y=data[anomaly_indices],
                                mode='markers', name='Anomalies',
                                marker=dict(color='red', size=8)), row=3, col=1)
    
    fig.update_layout(height=300 * n_subplots, title_text="Time Series Analysis Dashboard")
    return fig


if __name__ == "__main__":
    # Example usage
    config = PlotConfig()
    visualizer = TimeSeriesVisualizer(config)
    
    # Generate sample data
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    data = np.sin(t) + 0.1 * np.random.randn(100)
    
    # Basic plot
    visualizer.plot_time_series(data, title="Sample Time Series")
    
    # Forecast plot
    forecast = np.sin(t[-20:]) + 0.1 * np.random.randn(20)
    visualizer.plot_forecast(data, forecast, title="Sample Forecast")
    
    print("Visualization module loaded successfully")
