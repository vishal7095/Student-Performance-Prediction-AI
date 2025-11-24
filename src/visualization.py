"""
Visualization module for Student Performance Predictor
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime

from .utils import Config, get_feature_importance_color, format_percentage, format_score

logger = logging.getLogger(__name__)

class Visualizer:
    """Handle all visualizations for the project"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        self.output_dir = "output/visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def set_style(self):
        """Set consistent plotting style"""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
    
    def save_plot(self, fig, filename: str, dpi: int = 300) -> str:
        """Save plot to file"""
        filepath = os.path.join(self.output_dir, filename)
        
        if hasattr(fig, 'savefig'):
            # Matplotlib figure
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        else:
            # Plotly figure
            fig.write_image(filepath)
        
        logger.info(f"Plot saved: {filepath}")
        return filepath
    
    def create_data_distribution_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create distribution plots for all features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"Distribution of {col}" for col in numeric_columns],
            vertical_spacing=0.1
        )
        
        row, col = 1, 1
        for i, column in enumerate(numeric_columns):
            if i > 0 and i % 3 == 0:
                row += 1
                col = 1
            else:
                col = i % 3 + 1
            
            # Add histogram
            fig.add_trace(
                go.Histogram(x=df[column], name=column, nbinsx=30),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Feature Distributions",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            text=correlation_matrix.round(2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=600
        )
        
        return fig
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float]) -> go.Figure:
        """Create feature importance bar chart"""
        if not feature_importance:
            return self._create_empty_plot("No feature importance data available")
        
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # Create colors based on importance
        colors = [get_feature_importance_color(imp) for imp in importances]
        
        fig = go.Figure(data=[
            go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker_color=colors,
                text=[format_percentage(imp) for imp in importances],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
            xaxis_tickformat='.0%',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_prediction_accuracy_plot(self, actual_scores: List[float], 
                                      predicted_scores: List[float]) -> go.Figure:
        """Create prediction accuracy scatter plot"""
        fig = go.Figure()
        
        # Perfect prediction line
        max_score = max(max(actual_scores), max(predicted_scores))
        min_score = min(min(actual_scores), min(predicted_scores))
        
        fig.add_trace(go.Scatter(
            x=[min_score, max_score],
            y=[min_score, max_score],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='gray')
        ))
        
        # Actual predictions
        errors = [abs(a - p) for a, p in zip(actual_scores, predicted_scores)]
        
        fig.add_trace(go.Scatter(
            x=actual_scores,
            y=predicted_scores,
            mode='markers',
            name='Predictions',
            marker=dict(
                size=8,
                color=errors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Absolute Error')
            ),
            text=[f"Actual: {a}<br>Predicted: {p}<br>Error: {e:.1f}" 
                  for a, p, e in zip(actual_scores, predicted_scores, errors)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Prediction Accuracy",
            xaxis_title="Actual Scores",
            yaxis_title="Predicted Scores",
            width=700,
            height=600
        )
        
        return fig
    
    def create_performance_comparison_plot(self, model_metrics: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create model performance comparison plot"""
        if not model_metrics:
            return self._create_empty_plot("No model metrics available")
        
        models = list(model_metrics.keys())
        metrics = ['r2', 'mae', 'rmse']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['R² Score (Higher better)', 'MAE (Lower better)', 'RMSE (Lower better)'],
            shared_yaxes=True
        )
        
        for i, metric in enumerate(metrics):
            values = [model_metrics[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=values,
                    y=models,
                    orientation='h',
                    name=metric.upper(),
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto'
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_feedback_analysis_plot(self, feedback_stats: Dict[str, Any]) -> go.Figure:
        """Create feedback analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Feedback by User Type',
                'Feedback Ratings Distribution',
                'Prediction Accuracy Over Time',
                'Feedback Volume Over Time'
            ],
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Pie chart for user types
        if 'feedback_by_user_type' in feedback_stats:
            user_types = list(feedback_stats['feedback_by_user_type'].keys())
            counts = list(feedback_stats['feedback_by_user_type'].values())
            
            fig.add_trace(
                go.Pie(labels=user_types, values=counts, name="User Types"),
                row=1, col=1
            )
        
        # Histogram for ratings
        # This would need actual rating data - for now, placeholder
        
        # Prediction accuracy over time (placeholder)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name="Accuracy"),
            row=2, col=1
        )
        
        # Feedback volume over time (placeholder)
        fig.add_trace(
            go.Bar(x=[], y=[], name="Feedback Volume"),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Feedback Analysis Dashboard",
            height=600
        )
        
        return fig
    
    def create_student_profile_plot(self, student_data: Dict[str, Any], 
                                  prediction_result: Dict[str, Any]) -> go.Figure:
        """Create comprehensive student profile visualization"""
        # Radar chart for student attributes
        categories = ['Attendance', 'Study Hours', 'Previous Score', 
                     'Assignment Score', 'Participation']
        
        # Normalize values for radar chart
        normalized_values = [
            student_data.get('attendance', 0) / 100,
            student_data.get('study_hours', 0) / 10,
            student_data.get('previous_score', 0) / 100,
            student_data.get('assignment_score', 0) / 100,
            student_data.get('participation', 0) / 10
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],  # Close the circle
            theta=categories + [categories[0]],
            fill='toself',
            name='Student Profile',
            line=dict(color='blue')
        ))
        
        # Add average line for comparison
        avg_values = [0.7, 0.5, 0.7, 0.7, 0.6]  # Example averages
        fig.add_trace(go.Scatterpolar(
            r=avg_values + [avg_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Class Average',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Student Profile Comparison"
        )
        
        return fig
    
    def create_training_progress_plot(self, training_history: Dict[str, Any]) -> go.Figure:
        """Create training progress visualization"""
        if not training_history:
            return self._create_empty_plot("No training history available")
        
        # Extract metrics over time
        timestamps = []
        r2_scores = []
        mae_scores = []
        
        for timestamp, data in training_history.items():
            timestamps.append(timestamp)
            metrics = data.get('metrics', {})
            r2_scores.append(metrics.get('r2', 0))
            mae_scores.append(metrics.get('mae', 0))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['R² Score Over Time', 'MAE Over Time'],
            shared_xaxes=True
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=r2_scores, mode='lines+markers', name='R² Score'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=mae_scores, mode='lines+markers', name='MAE'),
            row=2, col=1
        )
        
        fig.update_layout(
            title_text="Model Training Progress",
            height=600
        )
        
        fig.update_yaxes(title_text="R² Score", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=2, col=1)
        fig.update_xaxes(title_text="Training Date", row=2, col=1)
        
        return fig
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        return fig
    
    def create_comprehensive_dashboard(self, data_processor, model_trainer, 
                                    predictor, feedback_system) -> go.Figure:
        """Create a comprehensive dashboard with multiple visualizations"""
        # Get data
        df = data_processor.load_processed_data()
        model_info = predictor.get_model_info()
        feedback_stats = feedback_system.get_feedback_statistics()
        
        # Create dashboard with multiple sections
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Data Distribution - Final Scores',
                'Feature Correlation Heatmap',
                'Model Feature Importance',
                'Feedback Distribution by User Type',
                'Prediction Accuracy',
                'Model Performance Over Time'
            ],
            specs=[
                [{"type": "histogram"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Data Distribution
        fig.add_trace(
            go.Histogram(x=df['final_score'], nbinsx=20, name='Final Scores'),
            row=1, col=1
        )
        
        # 2. Correlation Heatmap (simplified)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1
                ),
                row=1, col=2
            )
        
        # 3. Feature Importance
        feature_importance = model_info.get('feature_importance', {})
        if feature_importance:
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            
            fig.add_trace(
                go.Bar(x=importances, y=features, orientation='h', name='Importance'),
                row=2, col=1
            )
        
        # 4. Feedback Distribution
        if 'feedback_by_user_type' in feedback_stats:
            user_types = list(feedback_stats['feedback_by_user_type'].keys())
            counts = list(feedback_stats['feedback_by_user_type'].values())
            
            fig.add_trace(
                go.Pie(labels=user_types, values=counts, name="Feedback"),
                row=2, col=2
            )
        
        # 5. Prediction Accuracy (placeholder)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='markers', name='Accuracy'),
            row=3, col=1
        )
        
        # 6. Performance Over Time (placeholder)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Performance'),
            row=3, col=2
        )
        
        fig.update_layout(
            title_text="Student Performance Predictor - Comprehensive Dashboard",
            height=1000,
            showlegend=False
        )
        
        return fig

    def export_all_visualizations(self, data_processor, model_trainer, 
                                predictor, feedback_system) -> Dict[str, str]:
        """Export all visualizations to files"""
        exported_files = {}
        
        try:
            # Get data
            df = data_processor.load_processed_data()
            model_info = predictor.get_model_info()
            training_results = getattr(model_trainer, 'models', {})
            feedback_stats = feedback_system.get_feedback_statistics()
            
            # 1. Data distribution
            fig_dist = self.create_data_distribution_plot(df)
            exported_files['distribution'] = self.save_plot(fig_dist, 'data_distribution.png')
            
            # 2. Correlation heatmap
            fig_corr = self.create_correlation_heatmap(df)
            exported_files['correlation'] = self.save_plot(fig_corr, 'correlation_heatmap.png')
            
            # 3. Feature importance
            feature_importance = model_info.get('feature_importance', {})
            if feature_importance:
                fig_importance = self.create_feature_importance_plot(feature_importance)
                exported_files['feature_importance'] = self.save_plot(fig_importance, 'feature_importance.png')
            
            # 4. Model comparison
            if training_results:
                model_metrics = {}
                for name, result in training_results.items():
                    if result.get('model') is not None:
                        model_metrics[name] = result.get('metrics', {})
                
                fig_comparison = self.create_performance_comparison_plot(model_metrics)
                exported_files['model_comparison'] = self.save_plot(fig_comparison, 'model_comparison.png')
            
            # 5. Comprehensive dashboard
            fig_dashboard = self.create_comprehensive_dashboard(
                data_processor, model_trainer, predictor, feedback_system
            )
            exported_files['dashboard'] = self.save_plot(fig_dashboard, 'comprehensive_dashboard.png')
            
            logger.info(f"Exported {len(exported_files)} visualizations")
            
        except Exception as e:
            logger.error(f"Error exporting visualizations: {e}")
        
        return exported_files