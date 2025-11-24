#!/usr/bin/env python3
"""
Main application file for Student Performance Predictor
"""
import os
import sys
import logging
import argparse
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import Config, create_directories, PerformanceTracker
from data_processing import DataProcessor
from model_training import ModelTrainer
from prediction import Predictor
from feedback_system import FeedbackSystem
from visualization import Visualizer

logger = logging.getLogger(__name__)

class StudentPerformancePredictor:
    """Main application class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.data_processor = DataProcessor(config_path)
        self.model_trainer = ModelTrainer(config_path)
        self.predictor = Predictor(config_path)
        self.feedback_system = FeedbackSystem(config_path)
        self.visualizer = Visualizer(config_path)
        self.performance_tracker = PerformanceTracker()
        
        # Ensure directories exist
        create_directories()
    
    def initialize_system(self) -> bool:
        """Initialize the complete system"""
        try:
            logger.info("Initializing Student Performance Predictor System...")
            
            # Generate sample data if needed
            if not os.path.exists(self.config.get('data.raw_path')):
                logger.info("Generating sample data...")
                sample_data = self.data_processor.generate_sample_data()
                self.data_processor.save_raw_data(sample_data)
            
            # Process data
            processed_data = self.data_processor.load_processed_data()
            logger.info(f"Data processed: {len(processed_data)} students")
            
            # Train model if needed
            if not os.path.exists(self.config.get('model.save_path')):
                logger.info("Training initial model...")
                training_results = self.model_trainer.train_complete_pipeline()
                logger.info(f"Model trained: {training_results['best_model']} "
                           f"(R²: {training_results['best_metrics']['r2']:.3f})")
            else:
                logger.info("Loading existing model...")
                self.predictor.load_model()
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def train_new_model(self) -> Dict[str, Any]:
        """Train a new model from scratch"""
        try:
            logger.info("Starting new model training...")
            
            training_results = self.model_trainer.train_complete_pipeline()
            
            # Reload predictor with new model
            self.predictor.load_model()
            
            logger.info(f"New model training completed: {training_results['best_model']}")
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {'error': str(e)}
    
    def predict_student_score(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict score for a single student"""
        try:
            # Validate input
            is_valid, errors = self.predictor.validate_prediction_input(student_data)
            if not is_valid:
                return {'error': 'Invalid input data', 'details': errors}
            
            # Make prediction
            result = self.predictor.predict_single(student_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'error': str(e)}
    
    def predict_batch_scores(self, students_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict scores for multiple students"""
        try:
            results = self.predictor.predict_batch(students_data)
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [{'error': str(e)} for _ in students_data]
    
    def submit_feedback(self, user_type: str, user_id: str, student_id: str,
                       predicted_score: float, **kwargs) -> str:
        """Submit feedback for a prediction"""
        return self.feedback_system.submit_feedback(
            user_type, user_id, student_id, predicted_score, **kwargs
        )
    
    def retrain_with_feedback(self) -> Dict[str, Any]:
        """Retrain model incorporating feedback"""
        return self.feedback_system.retrain_model_with_feedback()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'system': {
                'initialized': True,
                'model_loaded': self.predictor.model is not None
            },
            'data': {
                'raw_data_exists': os.path.exists(self.config.get('data.raw_path')),
                'processed_data_exists': os.path.exists(self.config.get('data.processed_path'))
            },
            'model': self.predictor.get_model_info(),
            'feedback': self.feedback_system.get_feedback_statistics()
        }
        
        # Add data statistics if available
        try:
            processed_data = self.data_processor.load_processed_data()
            status['data']['statistics'] = self.data_processor.get_data_statistics(processed_data)
        except Exception as e:
            status['data']['statistics_error'] = str(e)
        
        return status
    
    def generate_reports(self) -> Dict[str, Any]:
        """Generate comprehensive reports"""
        reports = {}
        
        try:
            # Model performance report
            if hasattr(self.model_trainer, 'get_model_performance_report'):
                reports['model_performance'] = self.model_trainer.get_model_performance_report()
            
            # Feedback report
            reports['feedback_analysis'] = self.feedback_system.generate_feedback_report()
            
            # System status report
            reports['system_status'] = self.get_system_status()
            
            # Generate visualizations
            reports['visualizations'] = self.visualizer.export_all_visualizations(
                self.data_processor, self.model_trainer, self.predictor, self.feedback_system
            )
            
            logger.info("Generated comprehensive reports")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            reports['error'] = str(e)
        
        return reports
    
    def run_demo(self) -> None:
        """Run a demonstration of the system"""
        logger.info("Starting system demonstration...")
        
        try:
            # 1. Show system status
            status = self.get_system_status()
            print("\n" + "="*50)
            print("SYSTEM STATUS")
            print("="*50)
            print(f"Model: {status['model'].get('model_name', 'Not loaded')}")
            print(f"Training Date: {status['model'].get('training_date', 'Unknown')}")
            print(f"R² Score: {status['model'].get('performance_metrics', {}).get('r2', 'Unknown')}")
            print(f"Total Feedback: {status['feedback'].get('total_feedback', 0)}")
            
            # 2. Make sample predictions
            print("\n" + "="*50)
            print("SAMPLE PREDICTIONS")
            print("="*50)
            
            sample_students = [
                {
                    'name': 'Excellent Student',
                    'attendance': 95,
                    'study_hours': 8,
                    'previous_score': 90,
                    'assignment_score': 92,
                    'participation': 9
                },
                {
                    'name': 'Average Student',
                    'attendance': 80,
                    'study_hours': 5,
                    'previous_score': 75,
                    'assignment_score': 78,
                    'participation': 6
                },
                {
                    'name': 'Struggling Student',
                    'attendance': 65,
                    'study_hours': 2,
                    'previous_score': 55,
                    'assignment_score': 60,
                    'participation': 3
                }
            ]
            
            for student in sample_students:
                result = self.predict_student_score(student)
                if 'error' not in result:
                    print(f"\n{student['name']}:")
                    print(f"  Predicted Score: {result['predicted_score']}")
                    print(f"  Confidence: {result['confidence']:.1%}")
                    print(f"  Category: {result['performance_category']}")
                    print(f"  Top Recommendation: {result['recommendations'][0]}")
                else:
                    print(f"\n{student['name']}: Prediction failed - {result['error']}")
            
            # 3. Show feature importance
            print("\n" + "="*50)
            print("FEATURE IMPORTANCE")
            print("="*50)
            feature_importance = status['model'].get('feature_importance', {})
            for feature, importance in list(feature_importance.items())[:5]:
                print(f"  {feature}: {importance:.1%}")
            
            # 4. Feedback system demo
            print("\n" + "="*50)
            print("FEEDBACK SYSTEM")
            print("="*50)
            feedback_stats = status['feedback']
            print(f"Total Feedback: {feedback_stats['total_feedback']}")
            print(f"Feedback by User Type: {feedback_stats.get('feedback_by_user_type', {})}")
            
            if feedback_stats['total_feedback'] > 0:
                should_retrain, reason = self.feedback_system.should_retrain_model()
                print(f"Should Retrain: {should_retrain}")
                if should_retrain:
                    print(f"Reason: {reason}")
            
            print("\nDemonstration completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"Demonstration failed: {e}")

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(description='Student Performance Predictor')
    parser.add_argument('--init', action='store_true', help='Initialize system')
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--reports', action='store_true', help='Generate reports')
    parser.add_argument('--retrain', action='store_true', help='Retrain with feedback')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Create application instance
    app = StudentPerformancePredictor(args.config)
    
    if args.init:
        success = app.initialize_system()
        sys.exit(0 if success else 1)
    
    elif args.train:
        results = app.train_new_model()
        if 'error' in results:
            print(f"Training failed: {results['error']}")
            sys.exit(1)
        else:
            print(f"Training successful: {results['best_model']} "
                  f"(R²: {results['best_metrics']['r2']:.3f})")
    
    elif args.demo:
        app.run_demo()
    
    elif args.status:
        status = app.get_system_status()
        print("System Status:")
        print(f"  Model: {status['model'].get('model_name', 'Not loaded')}")
        print(f"  R² Score: {status['model'].get('performance_metrics', {}).get('r2', 'Unknown')}")
        print(f"  Data Points: {status['data'].get('statistics', {}).get('total_students', 'Unknown')}")
        print(f"  Feedback Count: {status['feedback'].get('total_feedback', 0)}")
    
    elif args.reports:
        reports = app.generate_reports()
        print("Reports generated:")
        for report_type in reports.keys():
            print(f"  - {report_type}")
    
    elif args.retrain:
        results = app.retrain_with_feedback()
        if results['retrained']:
            print("Model retrained successfully!")
            print(f"New R²: {results['new_metrics']['r2']:.3f}")
            if 'improvement' in results and 'r2' in results['improvement']:
                print(f"R² Improvement: {results['improvement']['r2']:+.3f}")
        else:
            print(f"Retraining not performed: {results['reason']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()