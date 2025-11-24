"""
Model training module for Student Performance Predictor
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import joblib
import os
from datetime import datetime
import warnings

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .utils import Config, save_model_history, PerformanceTracker
from .data_processing import DataProcessor

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.data_processor = DataProcessor(config_path)
        self.performance_tracker = PerformanceTracker()
        
        # Model paths
        self.model_save_path = self.config.get('model.save_path')
        self.scaler_save_path = self.config.get('model.scaler_path')
        self.feature_names_path = self.config.get('model.feature_names_path')
        
        # Training parameters
        self.test_size = self.config.get('model.training.test_size', 0.2)
        self.random_state = self.config.get('model.training.random_state', 42)
        self.cv_folds = self.config.get('model.training.cv_folds', 5)
        
        # Initialize models
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.feature_names = None
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare data for training"""
        df = self.data_processor.load_processed_data()
        X, y, feature_names = self.data_processor.prepare_training_data(df)
        
        self.feature_names = feature_names
        logger.info(f"Data prepared with {len(feature_names)} features")
        
        return X, y, feature_names
    
    def create_model_pipeline(self, model_name: str, model):
        """Create preprocessing pipeline for model"""
        if model_name in ['linear_regression', 'ridge', 'lasso', 'svr']:
            # Models that benefit from scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            # Tree-based models don't need scaling
            pipeline = Pipeline([
                ('model', model)
            ])
        
        return pipeline
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with their configurations"""
        
        model_configs = {
            'linear_regression': {
                'model': LinearRegression(**self.config.get('model.algorithms.linear_regression', {})),
                'param_grid': {
                    'model__fit_intercept': [True, False]
                }
            },
            'ridge': {
                'model': Ridge(random_state=self.random_state),
                'param_grid': {
                    'model__alpha': [0.1, 1.0, 10.0],
                    'model__fit_intercept': [True, False]
                }
            },
            'lasso': {
                'model': Lasso(random_state=self.random_state),
                'param_grid': {
                    'model__alpha': [0.1, 1.0, 10.0],
                    'model__fit_intercept': [True, False]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'param_grid': {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [5, 10, None],
                    'model__min_samples_split': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'param_grid': {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 4, 5]
                }
            },
            'svr': {
                'model': SVR(),
                'param_grid': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf'],
                    'model__gamma': ['scale', 'auto']
                }
            }
        }
        
        # Update with config values if available
        for name, config in model_configs.items():
            model_params = self.config.get(f'model.algorithms.{name}', {})
            if model_params:
                base_model = config['model']
                base_model.set_params(**model_params)
        
        return model_configs
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and evaluate their performance"""
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        model_configs = self.initialize_models()
        results = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Create pipeline
                pipeline = self.create_model_pipeline(model_name, config['model'])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred)
                
                # Store results
                results[model_name] = {
                    'model': pipeline,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'feature_importance': self.get_feature_importance(pipeline, model_name)
                }
                
                logger.info(f"{model_name} trained successfully. R²: {metrics['r2']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {
                    'model': None,
                    'metrics': {},
                    'error': str(e)
                }
        
        self.models = results
        return results
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'max_error': max(abs(y_true - y_pred)),
            'explained_variance': 1 - (np.var(y_true - y_pred) / np.var(y_true))
        }
    
    def get_feature_importance(self, model, model_name: str) -> Optional[Dict[str, float]]:
        """Get feature importance from model"""
        try:
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                importances = model.named_steps['model'].feature_importances_
            elif hasattr(model.named_steps['model'], 'coef_'):
                importances = abs(model.named_steps['model'].coef_)
            else:
                return None
            
            # Create feature importance dictionary
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = self.feature_names
            
            importance_dict = dict(zip(feature_names, importances))
            
            # Normalize to percentages
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"Could not get feature importance for {model_name}: {e}")
            return None
    
    def select_best_model(self) -> str:
        """Select the best model based on R² score"""
        best_score = -float('inf')
        best_model_name = None
        
        for model_name, result in self.models.items():
            if result['model'] is not None and 'r2' in result['metrics']:
                if result['metrics']['r2'] > best_score:
                    best_score = result['metrics']['r2']
                    best_model_name = model_name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]['model']
            self.best_model_name = best_model_name
            logger.info(f"Best model selected: {best_model_name} (R²: {best_score:.3f})")
        else:
            logger.error("No suitable model found!")
            raise ValueError("No model could be trained successfully")
        
        return best_model_name
    
    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                               model_name: str, model) -> Dict[str, float]:
        """Perform cross-validation for a model"""
        try:
            cv_scores = cross_val_score(
                model, X, y, 
                cv=self.cv_folds, 
                scoring='r2',
                n_jobs=-1
            )
            
            return {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed for {model_name}: {e}")
            return {}
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            model_name: str, pipeline, param_grid: Dict[str, List]) -> Any:
        """Perform hyperparameter tuning using GridSearchCV"""
        try:
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=self.cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logger.info(f"Best CV score for {model_name}: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed for {model_name}: {e}")
            return pipeline
    
    def save_best_model(self) -> None:
        """Save the best model and related artifacts"""
        if self.best_model is None:
            logger.error("No best model to save!")
            return
        
        # Save model
        joblib.dump(self.best_model, self.model_save_path)
        
        # Save feature names
        joblib.dump(self.feature_names, self.feature_names_path)
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'metrics': self.models[self.best_model_name]['metrics'],
            'feature_importance': self.models[self.best_model_name]['feature_importance']
        }
        
        model_info_path = self.model_save_path.replace('.pkl', '_info.pkl')
        joblib.dump(model_info, model_info_path)
        
        logger.info(f"Best model saved to {self.model_save_path}")
        logger.info(f"Model info saved to {model_info_path}")
    
    def load_model(self) -> bool:
        """Load trained model and artifacts"""
        try:
            if not os.path.exists(self.model_save_path):
                logger.warning("Model file not found")
                return False
            
            self.best_model = joblib.load(self.model_save_path)
            self.feature_names = joblib.load(self.feature_names_path)
            
            # Load model info
            model_info_path = self.model_save_path.replace('.pkl', '_info.pkl')
            if os.path.exists(model_info_path):
                model_info = joblib.load(model_info_path)
                self.best_model_name = model_info['model_name']
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def train_complete_pipeline(self) -> Dict[str, Any]:
        """Complete training pipeline from data loading to model saving"""
        logger.info("Starting complete training pipeline...")
        
        # Prepare data
        X, y, feature_names = self.prepare_data()
        
        # Train models
        training_results = self.train_models(X, y)
        
        # Select best model
        best_model_name = self.select_best_model()
        
        # Save model history
        timestamp = datetime.now().isoformat()
        save_model_history(
            best_model_name,
            training_results[best_model_name]['metrics'],
            feature_names,
            timestamp
        )
        
        # Track performance
        self.performance_tracker.track_performance(
            best_model_name,
            training_results[best_model_name]['metrics'],
            len(X),
            timestamp
        )
        
        # Save best model
        self.save_best_model()
        
        # Prepare results
        results = {
            'best_model': best_model_name,
            'best_metrics': training_results[best_model_name]['metrics'],
            'all_models': training_results,
            'feature_importance': training_results[best_model_name]['feature_importance'],
            'dataset_size': len(X),
            'training_date': timestamp
        }
        
        logger.info("Training pipeline completed successfully")
        return results
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.models:
            return {}
        
        report = {
            'best_model': self.best_model_name,
            'models_comparison': {},
            'feature_analysis': {},
            'recommendations': []
        }
        
        # Model comparison
        for model_name, result in self.models.items():
            if result['model'] is not None:
                report['models_comparison'][model_name] = result['metrics']
        
        # Feature importance analysis
        if self.best_model_name and self.best_model_name in self.models:
            feature_importance = self.models[self.best_model_name]['feature_importance']
            if feature_importance:
                report['feature_analysis'] = {
                    'most_important': sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True)[:3],
                    'least_important': sorted(feature_importance.items(), 
                                            key=lambda x: x[1])[:3]
                }
        
        # Recommendations
        best_metrics = self.models[self.best_model_name]['metrics']
        
        if best_metrics['r2'] > 0.8:
            report['recommendations'].append("Excellent model performance! The model can reliably predict student scores.")
        elif best_metrics['r2'] > 0.6:
            report['recommendations'].append("Good model performance. Consider collecting more data or additional features.")
        else:
            report['recommendations'].append("Model performance needs improvement. Review feature selection and data quality.")
        
        if best_metrics['mae'] > 8:
            report['recommendations'].append("High prediction error. Model may need more training data or feature engineering.")
        
        return report