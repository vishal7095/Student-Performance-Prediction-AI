"""
Feedback system for continuous model improvement
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime, timedelta
import json

from .utils import Config, generate_student_id
from .model_training import ModelTrainer
from .prediction import Predictor

logger = logging.getLogger(__name__)

class FeedbackSystem:
    """Handle feedback collection and model improvement"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.feedback_path = self.config.get('data.feedback_path')
        self.model_trainer = ModelTrainer(config_path)
        self.predictor = Predictor(config_path)
        
        # Feedback parameters
        self.min_feedback_for_retrain = self.config.get('feedback.min_feedback_for_retrain', 10)
        self.feedback_weights = self.config.get('feedback.feedback_weights', {})
        
        # Initialize feedback storage
        self._ensure_feedback_file_exists()
    
    def _ensure_feedback_file_exists(self) -> None:
        """Create feedback file if it doesn't exist"""
        if not os.path.exists(self.feedback_path):
            os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)
            empty_df = pd.DataFrame(columns=[
                'feedback_id', 'timestamp', 'user_type', 'user_id', 
                'student_id', 'predicted_score', 'actual_score',
                'feedback_rating', 'feedback_comment', 'model_version',
                'input_features', 'weight'
            ])
            empty_df.to_csv(self.feedback_path, index=False)
            logger.info(f"Created new feedback file: {self.feedback_path}")
    
    def submit_feedback(self, user_type: str, user_id: str, student_id: str,
                       predicted_score: float, actual_score: Optional[float] = None,
                       feedback_rating: Optional[int] = None,
                       feedback_comment: Optional[str] = None,
                       input_features: Optional[Dict[str, Any]] = None) -> str:
        """Submit feedback for a prediction"""
        
        feedback_id = generate_student_id(f"feedback_{user_id}", len(self.load_feedback()))
        
        # Calculate weight based on user type
        weight = self.feedback_weights.get(user_type, 1.0)
        
        # Create feedback record
        feedback_record = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'user_type': user_type,
            'user_id': user_id,
            'student_id': student_id,
            'predicted_score': predicted_score,
            'actual_score': actual_score,
            'feedback_rating': feedback_rating,
            'feedback_comment': feedback_comment,
            'model_version': self._get_current_model_version(),
            'input_features': json.dumps(input_features) if input_features else None,
            'weight': weight
        }
        
        # Save feedback
        self._save_feedback_record(feedback_record)
        
        logger.info(f"Feedback submitted by {user_type} {user_id} for student {student_id}")
        return feedback_id
    
    def _save_feedback_record(self, record: Dict[str, Any]) -> None:
        """Save a single feedback record to CSV"""
        try:
            # Load existing feedback
            feedback_df = self.load_feedback()
            
            # Append new record
            new_row = pd.DataFrame([record])
            updated_df = pd.concat([feedback_df, new_row], ignore_index=True)
            
            # Save updated feedback
            updated_df.to_csv(self.feedback_path, index=False)
            
        except Exception as e:
            logger.error(f"Error saving feedback record: {e}")
            # Fallback: append to file
            try:
                with open(self.feedback_path, 'a') as f:
                    f.write(','.join(str(record.get(col, '')) for col in [
                        'feedback_id', 'timestamp', 'user_type', 'user_id', 
                        'student_id', 'predicted_score', 'actual_score',
                        'feedback_rating', 'feedback_comment', 'model_version',
                        'input_features', 'weight'
                    ]) + '\n')
            except Exception as e2:
                logger.error(f"Fallback feedback save also failed: {e2}")
    
    def load_feedback(self) -> pd.DataFrame:
        """Load all feedback records"""
        try:
            df = pd.read_csv(self.feedback_path)
            
            # Convert JSON string back to dict for input_features
            if 'input_features' in df.columns:
                df['input_features'] = df['input_features'].apply(
                    lambda x: json.loads(x) if pd.notna(x) and x != '' else {}
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            return pd.DataFrame()
    
    def _get_current_model_version(self) -> str:
        """Get current model version identifier"""
        try:
            model_info = self.predictor.get_model_info()
            return f"{model_info.get('model_name', 'unknown')}_{model_info.get('training_date', 'unknown')}"
        except:
            return "unknown"
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        feedback_df = self.load_feedback()
        
        if feedback_df.empty:
            return {
                'total_feedback': 0,
                'feedback_by_user_type': {},
                'average_rating': 0,
                'feedback_with_actual_scores': 0
            }
        
        stats = {
            'total_feedback': len(feedback_df),
            'feedback_by_user_type': feedback_df['user_type'].value_counts().to_dict(),
            'average_rating': feedback_df['feedback_rating'].mean() if 'feedback_rating' in feedback_df else 0,
            'feedback_with_actual_scores': feedback_df['actual_score'].notna().sum(),
            'earliest_feedback': feedback_df['timestamp'].min(),
            'latest_feedback': feedback_df['timestamp'].max()
        }
        
        # Calculate prediction accuracy for records with actual scores
        if stats['feedback_with_actual_scores'] > 0:
            valid_feedback = feedback_df[feedback_df['actual_score'].notna()]
            errors = abs(valid_feedback['predicted_score'] - valid_feedback['actual_score'])
            stats['mean_absolute_error'] = errors.mean()
            stats['accuracy_within_5_points'] = (errors <= 5).sum() / len(errors)
        
        return stats
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in feedback data"""
        feedback_df = self.load_feedback()
        
        if feedback_df.empty:
            return {}
        
        analysis = {
            'common_issues': {},
            'user_sentiment': {},
            'model_performance_trends': {}
        }
        
        # Analyze feedback comments
        if 'feedback_comment' in feedback_df.columns:
            comments = feedback_df[feedback_df['feedback_comment'].notna()]['feedback_comment']
            
            # Simple keyword analysis (in a real system, use NLP)
            issues = {
                'over_prediction': comments.str.contains('over|high|too much', case=False).sum(),
                'under_prediction': comments.str.contains('under|low|too little', case=False).sum(),
                'inaccurate': comments.str.contains('wrong|incorrect|inaccurate', case=False).sum(),
                'good_prediction': comments.str.contains('good|accurate|correct|nice', case=False).sum()
            }
            
            analysis['common_issues'] = issues
        
        # Analyze by user type
        if 'user_type' in feedback_df.columns and 'feedback_rating' in feedback_df.columns:
            sentiment_by_type = feedback_df.groupby('user_type')['feedback_rating'].agg(['mean', 'count'])
            analysis['user_sentiment'] = sentiment_by_type.to_dict()
        
        # Performance trends over time
        if 'timestamp' in feedback_df.columns and 'actual_score' in feedback_df.columns:
            time_based = feedback_df[feedback_df['actual_score'].notna()].copy()
            if not time_based.empty:
                time_based['date'] = pd.to_datetime(time_based['timestamp']).dt.date
                daily_accuracy = time_based.groupby('date').apply(
                    lambda x: (abs(x['predicted_score'] - x['actual_score']) <= 5).mean()
                )
                analysis['model_performance_trends'] = {
                    'daily_accuracy': daily_accuracy.to_dict(),
                    'overall_trend': 'improving' if len(daily_accuracy) > 1 and 
                                     daily_accuracy.iloc[-1] > daily_accuracy.iloc[0] else 'stable'
                }
        
        return analysis
    
    def should_retrain_model(self) -> Tuple[bool, str]:
        """Determine if model should be retrained based on feedback"""
        stats = self.get_feedback_statistics()
        
        reasons = []
        
        # Check minimum feedback count
        if stats['total_feedback'] < self.min_feedback_for_retrain:
            return False, f"Not enough feedback (have {stats['total_feedback']}, need {self.min_feedback_for_retrain})"
        
        # Check if we have sufficient actual scores for evaluation
        if stats['feedback_with_actual_scores'] < self.min_feedback_for_retrain // 2:
            reasons.append("Insufficient actual scores for proper evaluation")
        
        # Check performance metrics
        if 'mean_absolute_error' in stats and stats['mean_absolute_error'] > 8:
            reasons.append(f"High prediction error (MAE: {stats['mean_absolute_error']:.1f})")
        
        if 'accuracy_within_5_points' in stats and stats['accuracy_within_5_points'] < 0.6:
            reasons.append(f"Low accuracy within 5 points ({stats['accuracy_within_5_points']:.1%})")
        
        # Check feedback ratings
        if stats['average_rating'] < 3 and stats['total_feedback'] >= 10:
            reasons.append(f"Low average feedback rating ({stats['average_rating']:.1f})")
        
        if reasons:
            return True, "; ".join(reasons)
        else:
            return False, "Model performance is satisfactory"
    
    def prepare_retraining_data(self) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for model retraining including feedback"""
        try:
            # Load original processed data
            original_df = self.model_trainer.data_processor.load_processed_data()
            
            # Load feedback with actual scores
            feedback_df = self.load_feedback()
            valid_feedback = feedback_df[
                feedback_df['actual_score'].notna() & 
                feedback_df['input_features'].notna()
            ].copy()
            
            if valid_feedback.empty:
                logger.info("No valid feedback data for retraining")
                return None
            
            # Convert input features from JSON strings
            features_data = []
            for _, row in valid_feedback.iterrows():
                try:
                    if isinstance(row['input_features'], str):
                        features = json.loads(row['input_features'])
                    else:
                        features = row['input_features']
                    
                    # Add actual score
                    features['final_score'] = row['actual_score']
                    features_data.append(features)
                    
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Invalid input features in feedback {row['feedback_id']}: {e}")
                    continue
            
            if not features_data:
                return None
            
            # Create feedback DataFrame
            feedback_data_df = pd.DataFrame(features_data)
            
            # Combine with original data (weighted by feedback weight)
            combined_df = self._combine_datasets(original_df, feedback_data_df, valid_feedback)
            
            # Prepare features and target
            X = combined_df[self.predictor.feature_names]
            y = combined_df['final_score']
            
            logger.info(f"Prepared retraining data: {len(combined_df)} samples "
                       f"({len(original_df)} original + {len(valid_feedback)} feedback)")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing retraining data: {e}")
            return None
    
    def _combine_datasets(self, original_df: pd.DataFrame, feedback_df: pd.DataFrame, 
                         feedback_metadata: pd.DataFrame) -> pd.DataFrame:
        """Combine original data with feedback data using weights"""
        # For simplicity, we'll just concatenate
        # In a more sophisticated implementation, you might weight the feedback data
        combined_df = pd.concat([original_df, feedback_df], ignore_index=True)
        
        # Remove duplicates based on feature set (simplified)
        combined_df = combined_df.drop_duplicates(
            subset=self.predictor.feature_names, 
            keep='last'
        )
        
        return combined_df
    
    def retrain_model_with_feedback(self) -> Dict[str, Any]:
        """Retrain model incorporating feedback data"""
        logger.info("Starting model retraining with feedback...")
        
        # Check if retraining is needed
        should_retrain, reason = self.should_retrain_model()
        if not should_retrain:
            return {
                'retrained': False,
                'reason': reason,
                'new_metrics': None
            }
        
        try:
            # Prepare retraining data
            retraining_data = self.prepare_retraining_data()
            if retraining_data is None:
                return {
                    'retrained': False,
                    'reason': "No valid feedback data for retraining",
                    'new_metrics': None
                }
            
            X, y = retraining_data
            
            # Retrain models
            retrain_results = self.model_trainer.train_models(X, y)
            best_model_name = self.model_trainer.select_best_model()
            
            # Compare with previous performance
            old_metrics = self.predictor.get_model_info().get('performance_metrics', {})
            new_metrics = retrain_results[best_model_name]['metrics']
            
            # Save new model
            self.model_trainer.save_best_model()
            
            # Reload predictor with new model
            self.predictor.load_model()
            
            # Log retraining results
            improvement = {}
            if old_metrics and 'r2' in old_metrics and 'r2' in new_metrics:
                improvement['r2'] = new_metrics['r2'] - old_metrics['r2']
            
            if old_metrics and 'mae' in old_metrics and 'mae' in new_metrics:
                improvement['mae'] = old_metrics['mae'] - new_metrics['mae']  # Lower is better
            
            result = {
                'retrained': True,
                'reason': reason,
                'best_model': best_model_name,
                'old_metrics': old_metrics,
                'new_metrics': new_metrics,
                'improvement': improvement,
                'training_samples': len(X),
                'feedback_samples': len(self.load_feedback())
            }
            
            logger.info(f"Model retraining completed. R² improvement: {improvement.get('r2', 0):.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {
                'retrained': False,
                'reason': f"Retraining failed: {str(e)}",
                'new_metrics': None
            }
    
    def generate_feedback_report(self) -> Dict[str, Any]:
        """Generate comprehensive feedback analysis report"""
        stats = self.get_feedback_statistics()
        patterns = self.analyze_feedback_patterns()
        should_retrain, retrain_reason = self.should_retrain_model()
        
        report = {
            'summary': {
                'total_feedback': stats['total_feedback'],
                'feedback_period': {
                    'start': stats.get('earliest_feedback', 'N/A'),
                    'end': stats.get('latest_feedback', 'N/A')
                },
                'model_performance': {
                    'mean_absolute_error': stats.get('mean_absolute_error', 'N/A'),
                    'accuracy_within_5_points': stats.get('accuracy_within_5_points', 'N/A')
                }
            },
            'user_engagement': stats.get('feedback_by_user_type', {}),
            'feedback_analysis': patterns,
            'model_health': {
                'should_retrain': should_retrain,
                'retrain_reason': retrain_reason,
                'feedback_sufficiency': stats['total_feedback'] >= self.min_feedback_for_retrain
            },
            'recommendations': self._generate_feedback_recommendations(stats, patterns)
        }
        
        return report
    
    def _generate_feedback_recommendations(self, stats: Dict[str, Any], 
                                         patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on feedback analysis"""
        recommendations = []
        
        # Feedback quantity recommendations
        if stats['total_feedback'] < 20:
            recommendations.append("Encourage more users to provide feedback to improve model accuracy")
        
        if stats['feedback_with_actual_scores'] < 10:
            recommendations.append("Request users to provide actual scores when available for better model evaluation")
        
        # Performance-based recommendations
        if 'mean_absolute_error' in stats and stats['mean_absolute_error'] > 10:
            recommendations.append("Consider collecting additional features that might improve prediction accuracy")
        
        # User engagement recommendations
        user_types = stats.get('feedback_by_user_type', {})
        if 'teacher' not in user_types or user_types['teacher'] < 5:
            recommendations.append("Increase teacher engagement for more expert feedback")
        
        return recommendations