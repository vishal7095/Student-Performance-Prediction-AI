"""
Prediction module for Student Performance Predictor
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import joblib
from datetime import datetime

from .utils import Config
from .model_training import ModelTrainer

logger = logging.getLogger(__name__)

class Predictor:
    """Handle predictions using trained models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.model_trainer = ModelTrainer(config_path)
        self.model = None
        self.feature_names = None
        self.model_info = None
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            success = self.model_trainer.load_model()
            if success:
                self.model = self.model_trainer.best_model
                self.feature_names = self.model_trainer.feature_names
                
                # Load model info
                model_info_path = self.config.get('model.save_path').replace('.pkl', '_info.pkl')
                if joblib.__version__ >= '1.2':
                    self.model_info = joblib.load(model_info_path)
                else:
                    self.model_info = joblib.load(model_info_path)
                
                logger.info("Predictor initialized successfully")
                return True
            else:
                logger.warning("No trained model found. Please train a model first.")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
            return False
    
    def prepare_input_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for prediction"""
        try:
            # Create feature dictionary with default values
            features = {}
            
            for feature in self.feature_names:
                if feature in input_data:
                    features[feature] = [input_data[feature]]
                else:
                    # Set default value based on feature type
                    if 'attendance' in feature:
                        features[feature] = [85.0]  # Default attendance
                    elif 'study' in feature:
                        features[feature] = [5.0]   # Default study hours
                    elif 'score' in feature:
                        features[feature] = [75.0]  # Default score
                    elif 'participation' in feature:
                        features[feature] = [6]     # Default participation
                    else:
                        features[feature] = [0.0]   # Default for other features
            
            # Create DataFrame with correct feature order
            df = pd.DataFrame(features)[self.feature_names]
            
            logger.debug(f"Prepared features for prediction: {df.iloc[0].to_dict()}")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing input features: {e}")
            raise ValueError(f"Feature preparation failed: {e}")
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict final score for a single student"""
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        try:
            # Prepare features
            features_df = self.prepare_input_features(input_data)
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            
            # Ensure prediction is within reasonable bounds
            prediction = max(0, min(100, prediction))
            
            # Calculate confidence interval (simplified)
            confidence = self._calculate_confidence(prediction, features_df)
            
            # Get feature contributions
            feature_contributions = self._analyze_feature_contributions(features_df)
            
            # Performance category
            performance_category = self._get_performance_category(prediction)
            
            # Recommendations
            recommendations = self._generate_recommendations(
                input_data, prediction, feature_contributions
            )
            
            result = {
                'predicted_score': round(prediction, 1),
                'confidence': confidence,
                'performance_category': performance_category,
                'feature_contributions': feature_contributions,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_trainer.best_model_name
            }
            
            logger.info(f"Prediction completed: {prediction:.1f} (Confidence: {confidence:.1%})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction error: {e}")
    
    def predict_batch(self, students_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict scores for multiple students"""
        results = []
        
        for i, student_data in enumerate(students_data):
            try:
                result = self.predict_single(student_data)
                result['student_index'] = i
                if 'name' in student_data:
                    result['student_name'] = student_data['name']
                if 'student_id' in student_data:
                    result['student_id'] = student_data['student_id']
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Prediction failed for student {i}: {e}")
                results.append({
                    'student_index': i,
                    'error': str(e),
                    'student_name': student_data.get('name', 'Unknown'),
                    'student_id': student_data.get('student_id', 'Unknown')
                })
        
        logger.info(f"Batch prediction completed for {len(students_data)} students")
        return results
    
    def _calculate_confidence(self, prediction: float, features_df: pd.DataFrame) -> float:
        """Calculate prediction confidence (simplified implementation)"""
        try:
            # Base confidence on how extreme the prediction is
            # Predictions near the middle range (60-80) are more reliable
            if 60 <= prediction <= 80:
                base_confidence = 0.85
            elif 50 <= prediction < 60 or 80 < prediction <= 90:
                base_confidence = 0.75
            else:
                base_confidence = 0.65
            
            # Adjust based on input feature quality
            feature_quality = self._assess_feature_quality(features_df.iloc[0])
            confidence = base_confidence * feature_quality
            
            return min(0.95, max(0.5, confidence))
            
        except Exception:
            return 0.7  # Default confidence
    
    def _assess_feature_quality(self, features: pd.Series) -> float:
        """Assess quality of input features"""
        quality_score = 1.0
        
        # Check for extreme values
        for feature, value in features.items():
            if 'attendance' in feature and (value < 50 or value > 100):
                quality_score *= 0.8
            elif 'study' in feature and (value < 1 or value > 12):
                quality_score *= 0.8
            elif 'score' in feature and (value < 0 or value > 100):
                quality_score *= 0.7
            elif 'participation' in feature and (value < 1 or value > 10):
                quality_score *= 0.9
        
        return quality_score
    
    def _analyze_feature_contributions(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze how much each feature contributes to the prediction"""
        try:
            contributions = {}
            
            if hasattr(self.model.named_steps['model'], 'feature_importances_'):
                # Tree-based models
                importances = self.model.named_steps['model'].feature_importances_
                for feature, importance in zip(self.feature_names, importances):
                    contributions[feature] = float(importance * 100)  # as percentage
            
            elif hasattr(self.model.named_steps['model'], 'coef_'):
                # Linear models
                coefficients = self.model.named_steps['model'].coef_
                feature_values = features_df.iloc[0].values
                
                # Calculate contribution as coef * value
                for i, (feature, coef, value) in enumerate(zip(self.feature_names, coefficients, feature_values)):
                    contribution = coef * value
                    contributions[feature] = float(contribution)
                
                # Normalize to percentages
                total_contribution = sum(abs(c) for c in contributions.values())
                if total_contribution > 0:
                    contributions = {k: (abs(v) / total_contribution) * 100 
                                   for k, v in contributions.items()}
            
            else:
                # Fallback: equal contributions
                equal_share = 100.0 / len(self.feature_names)
                for feature in self.feature_names:
                    contributions[feature] = equal_share
            
            # Sort by contribution
            contributions = dict(sorted(contributions.items(), 
                                      key=lambda x: x[1], reverse=True))
            
            return contributions
            
        except Exception as e:
            logger.warning(f"Feature contribution analysis failed: {e}")
            return {feature: 100.0/len(self.feature_names) for feature in self.feature_names}
    
    def _get_performance_category(self, score: float) -> str:
        """Categorize performance based on score"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Satisfactory"
        elif score >= 50:
            return "Needs Improvement"
        else:
            return "At Risk"
    
    def _generate_recommendations(self, input_data: Dict[str, Any], 
                                prediction: float, 
                                contributions: Dict[str, float]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Attendance-based recommendations
        attendance = input_data.get('attendance', 85)
        if attendance < 75:
            recommendations.append("Consider improving attendance to boost performance")
        elif attendance > 95:
            recommendations.append("Excellent attendance! Maintain this consistency")
        
        # Study hours recommendations
        study_hours = input_data.get('study_hours', 5)
        if study_hours < 3:
            recommendations.append("Increase study hours to at least 4-5 hours daily")
        elif study_hours > 8:
            recommendations.append("Good study discipline! Ensure you're taking breaks")
        
        # Previous score recommendations
        previous_score = input_data.get('previous_score', 75)
        if previous_score < 60:
            recommendations.append("Focus on building foundational knowledge from previous topics")
        elif prediction > previous_score + 10:
            recommendations.append("Great improvement potential! Current efforts are paying off")
        
        # Participation recommendations
        participation = input_data.get('participation', 6)
        if participation < 5:
            recommendations.append("Increase class participation to enhance learning engagement")
        
        # Performance-based recommendations
        if prediction < 60:
            top_contributor = next(iter(contributions.keys()))
            recommendations.append(f"Focus on improving {top_contributor.replace('_', ' ')} for maximum impact")
        
        # General recommendations
        if len(recommendations) < 3:
            recommendations.extend([
                "Maintain consistent study schedule",
                "Review assignments and tests regularly",
                "Seek help early if struggling with topics"
            ])
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def validate_prediction_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data for prediction"""
        errors = []
        
        # Check required features
        for feature in self.feature_names:
            if feature not in input_data:
                errors.append(f"Missing required feature: {feature}")
        
        # Validate numeric ranges
        numeric_checks = {
            'attendance': (0, 100),
            'study_hours': (0, 24),
            'previous_score': (0, 100),
            'assignment_score': (0, 100),
            'participation': (1, 10)
        }
        
        for feature, (min_val, max_val) in numeric_checks.items():
            if feature in input_data:
                try:
                    value = float(input_data[feature])
                    if not (min_val <= value <= max_val):
                        errors.append(f"{feature} must be between {min_val} and {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"{feature} must be a numeric value")
        
        return len(errors) == 0, errors
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model_info is None:
            return {}
        
        return {
            'model_name': self.model_info.get('model_name', 'Unknown'),
            'feature_names': self.model_info.get('feature_names', []),
            'training_date': self.model_info.get('training_date', 'Unknown'),
            'performance_metrics': self.model_info.get('metrics', {}),
            'feature_importance': self.model_info.get('feature_importance', {})
        }
    
    def compare_with_actual(self, predictions: List[Dict[str, Any]], 
                          actual_scores: List[float]) -> Dict[str, Any]:
        """Compare predictions with actual scores"""
        if len(predictions) != len(actual_scores):
            raise ValueError("Predictions and actual scores must have the same length")
        
        predicted_scores = [p['predicted_score'] for p in predictions if 'predicted_score' in p]
        actual_scores_valid = [a for i, a in enumerate(actual_scores) 
                             if 'predicted_score' in predictions[i]]
        
        if not predicted_scores:
            return {}
        
        errors = [abs(p - a) for p, a in zip(predicted_scores, actual_scores_valid)]
        
        return {
            'mean_absolute_error': np.mean(errors),
            'max_error': max(errors),
            'accuracy_within_5_points': sum(e <= 5 for e in errors) / len(errors),
            'accuracy_within_10_points': sum(e <= 10 for e in errors) / len(errors),
            'perfect_predictions': sum(e == 0 for e in errors),
            'total_comparisons': len(predicted_scores)
        }