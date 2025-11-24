#!/usr/bin/env python3
"""
Streamlit web application for Student Performance Predictor
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import Config
from data_processing import DataProcessor
from model_training import ModelTrainer
from prediction import Predictor
from feedback_system import FeedbackSystem
from visualization import Visualizer

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = False
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'feedback_system' not in st.session_state:
    st.session_state.feedback_system = None

def initialize_app():
    """Initialize the application components"""
    try:
        config = Config("config/config.yaml")
        st.session_state.data_processor = DataProcessor("config/config.yaml")
        st.session_state.model_trainer = ModelTrainer("config/config.yaml")
        st.session_state.predictor = Predictor("config/config.yaml")
        st.session_state.feedback_system = FeedbackSystem("config/config.yaml")
        st.session_state.visualizer = Visualizer("config/config.yaml")
        st.session_state.app_initialized = True
        
        # Load data
        st.session_state.data = st.session_state.data_processor.load_processed_data()
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        return False

def main():
    """Main application function"""
    st.title("🎓 Student Performance Predictor")
    st.markdown("""
    This AI-powered system predicts student final scores based on academic and behavioral factors,
    and continuously improves through user feedback.
    """)
    
    # Initialize app if not already done
    if not st.session_state.app_initialized:
        with st.spinner("Initializing application..."):
            if not initialize_app():
                st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "🔮 Prediction", "📊 Analysis", "🔄 Feedback", "⚙️ System"]
    )
    
    # Route to selected page
    if app_mode == "🏠 Dashboard":
        show_dashboard()
    elif app_mode == "🔮 Prediction":
        show_prediction()
    elif app_mode == "📊 Analysis":
        show_analysis()
    elif app_mode == "🔄 Feedback":
        show_feedback()
    elif app_mode == "⚙️ System":
        show_system()

def show_dashboard():
    """Show main dashboard"""
    st.header("📊 System Dashboard")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            model_info = st.session_state.predictor.get_model_info()
            st.metric(
                "Model Status",
                "Loaded" if st.session_state.predictor.model else "Not Loaded",
                delta=model_info.get('model_name', 'Unknown')
            )
        except:
            st.metric("Model Status", "Error", delta="Check configuration")
    
    with col2:
        try:
            data_stats = st.session_state.data_processor.get_data_statistics(st.session_state.data)
            st.metric(
                "Students in Database",
                data_stats.get('total_students', 0)
            )
        except:
            st.metric("Students in Database", "Error")
    
    with col3:
        try:
            feedback_stats = st.session_state.feedback_system.get_feedback_statistics()
            st.metric(
                "Feedback Collected",
                feedback_stats.get('total_feedback', 0)
            )
        except:
            st.metric("Feedback Collected", "Error")
    
    with col4:
        try:
            metrics = model_info.get('performance_metrics', {})
            st.metric(
                "Model R² Score",
                f"{metrics.get('r2', 0):.3f}" if metrics else "N/A"
            )
        except:
            st.metric("Model R² Score", "Error")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Retrain Model", use_container_width=True):
            with st.spinner("Retraining model..."):
                result = st.session_state.feedback_system.retrain_model_with_feedback()
                if result['retrained']:
                    st.success("Model retrained successfully!")
                    st.session_state.predictor.load_model()  # Reload model
                else:
                    st.warning(f"Retraining not performed: {result['reason']}")
    
    with col2:
        if st.button("📈 Generate Reports", use_container_width=True):
            with st.spinner("Generating reports..."):
                # This would generate and display reports
                st.info("Report generation would be implemented here")
    
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Refreshing data..."):
                st.session_state.data = st.session_state.data_processor.load_processed_data()
                st.success("Data refreshed!")
    
    # Recent predictions placeholder
    st.subheader("Recent Activity")
    st.info("""
    **Recent Predictions:** No recent predictions to show.  
    **System Health:** All systems operational.  
    **Feedback Status:** Ready to collect user feedback.
    """)
    
    # Data overview
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sample Data (First 10 students)**")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
    
    with col2:
        st.write("**Data Statistics**")
        if 'data_stats' not in locals():
            data_stats = st.session_state.data_processor.get_data_statistics(st.session_state.data)
        
        stats_df = pd.DataFrame([
            {"Metric": "Total Students", "Value": data_stats.get('total_students', 0)},
            {"Metric": "Average Final Score", "Value": f"{data_stats.get('final_score', {}).get('mean', 0):.1f}"},
            {"Metric": "Average Attendance", "Value": f"{data_stats.get('attendance', {}).get('mean', 0):.1f}%"},
            {"Metric": "Average Study Hours", "Value": f"{data_stats.get('study_hours', {}).get('mean', 0):.1f}"}
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

def show_prediction():
    """Show prediction interface"""
    st.header("🔮 Student Performance Prediction")
    
    # Prediction type selection
    pred_type = st.radio(
        "Prediction Type",
        ["Single Student", "Batch Students"],
        horizontal=True
    )
    
    if pred_type == "Single Student":
        show_single_prediction()
    else:
        show_batch_prediction()

def show_single_prediction():
    """Show single student prediction interface"""
    st.subheader("Single Student Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Student Information**")
            name = st.text_input("Student Name", placeholder="e.g., John Smith")
            student_id = st.text_input("Student ID (Optional)", placeholder="e.g., STU001")
            
            st.write("**Academic Metrics**")
            previous_score = st.slider("Previous Exam Score", 0, 100, 75)
            assignment_score = st.slider("Assignment Score", 0, 100, 75)
        
        with col2:
            st.write("**Behavioral Metrics**")
            attendance = st.slider("Attendance (%)", 0, 100, 85)
            study_hours = st.slider("Study Hours per Day", 0.0, 10.0, 5.0, 0.5)
            participation = st.slider("Class Participation (1-10)", 1, 10, 6)
        
        # Submit button
        submitted = st.form_submit_button("Predict Final Score", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                'attendance': attendance,
                'study_hours': study_hours,
                'previous_score': previous_score,
                'assignment_score': assignment_score,
                'participation': participation
            }
            
            # Validate input
            is_valid, errors = st.session_state.predictor.validate_prediction_input(input_data)
            
            if not is_valid:
                st.error("Invalid input data:")
                for error in errors:
                    st.write(f"- {error}")
            else:
                # Make prediction
                with st.spinner("Analyzing student data..."):
                    result = st.session_state.predictor.predict_single(input_data)
                
                if 'error' in result:
                    st.error(f"Prediction failed: {result['error']}")
                else:
                    # Display results
                    st.success("Prediction completed successfully!")
                    
                    # Results in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Predicted Final Score",
                            f"{result['predicted_score']}",
                            delta=result['performance_category']
                        )
                    
                    with col2:
                        st.metric(
                            "Prediction Confidence",
                            f"{result['confidence']:.1%}"
                        )
                    
                    with col3:
                        st.metric(
                            "Model Used",
                            result['model_used'].replace('_', ' ').title()
                        )
                    
                    # Feature contributions
                    st.subheader("Feature Analysis")
                    features_df = pd.DataFrame(
                        list(result['feature_contributions'].items()),
                        columns=['Feature', 'Contribution (%)']
                    )
                    features_df['Contribution (%)'] = features_df['Contribution (%)'].round(2)
                    st.dataframe(features_df, use_container_width=True, hide_index=True)
                    
                    # Recommendations
                    st.subheader("Personalized Recommendations")
                    for i, recommendation in enumerate(result['recommendations'][:5], 1):
                        st.write(f"{i}. {recommendation}")
                    
                    # Feedback section
                    st.subheader("Provide Feedback")
                    feedback_col1, feedback_col2 = st.columns(2)
                    
                    with feedback_col1:
                        actual_score = st.number_input(
                            "Actual Score (if known)",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(result['predicted_score']),
                            step=0.5
                        )
                    
                    with feedback_col2:
                        feedback_rating = st.slider(
                            "How accurate was this prediction?",
                            1, 5, 3,
                            help="1 = Very inaccurate, 5 = Very accurate"
                        )
                    
                    feedback_comment = st.text_area(
                        "Additional comments (optional)",
                        placeholder="Any additional feedback about this prediction..."
                    )
                    
                    if st.button("Submit Feedback", type="primary"):
                        if name or student_id:
                            feedback_id = st.session_state.feedback_system.submit_feedback(
                                user_type="web_user",
                                user_id="anonymous",
                                student_id=student_id or name or "unknown",
                                predicted_score=result['predicted_score'],
                                actual_score=actual_score if actual_score != result['predicted_score'] else None,
                                feedback_rating=feedback_rating,
                                feedback_comment=feedback_comment if feedback_comment else None,
                                input_features=input_data
                            )
                            st.success("Thank you for your feedback! It will help improve the model.")
                        else:
                            st.warning("Please provide at least a name or student ID for feedback tracking")

def show_batch_prediction():
    """Show batch prediction interface"""
    st.subheader("Batch Student Prediction")
    
    st.info("""
    **Batch Prediction Instructions:**
    - Upload a CSV file with columns: attendance, study_hours, previous_score, assignment_score, participation
    - Alternatively, use the template below to manually enter multiple students
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("**Uploaded Data Preview:**")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            if st.button("Process Batch Prediction", type="primary"):
                with st.spinner("Processing batch prediction..."):
                    # Convert to list of dictionaries
                    students_list = batch_data.to_dict('records')
                    results = st.session_state.predictor.predict_batch(students_list)
                
                # Display results
                results_df = pd.DataFrame(results)
                st.success(f"Batch prediction completed for {len(results)} students!")
                
                # Show results table
                st.write("**Prediction Results:**")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV",
                    csv,
                    "batch_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        # Manual batch input
        st.write("**Manual Batch Entry**")
        num_students = st.number_input("Number of students", 1, 50, 3)
        
        students_data = []
        for i in range(num_students):
            st.write(f"**Student {i+1}**")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(f"Name {i+1}", key=f"name_{i}")
                attendance = st.slider(f"Attendance {i+1}", 0, 100, 85, key=f"att_{i}")
                study_hours = st.slider(f"Study Hours {i+1}", 0.0, 10.0, 5.0, 0.5, key=f"study_{i}")
            
            with col2:
                previous_score = st.slider(f"Previous Score {i+1}", 0, 100, 75, key=f"prev_{i}")
                assignment_score = st.slider(f"Assignment Score {i+1}", 0, 100, 75, key=f"assign_{i}")
                participation = st.slider(f"Participation {i+1}", 1, 10, 6, key=f"part_{i}")
            
            students_data.append({
                'name': name,
                'attendance': attendance,
                'study_hours': study_hours,
                'previous_score': previous_score,
                'assignment_score': assignment_score,
                'participation': participation
            })
        
        if st.button("Predict Batch Scores", type="primary"):
            with st.spinner("Processing predictions..."):
                results = st.session_state.predictor.predict_batch(students_data)
            
            # Display results
            display_results = []
            for i, result in enumerate(results):
                if 'error' not in result:
                    display_results.append({
                        'Student': students_data[i].get('name', f'Student {i+1}'),
                        'Predicted Score': result['predicted_score'],
                        'Confidence': f"{result['confidence']:.1%}",
                        'Category': result['performance_category'],
                        'Top Recommendation': result['recommendations'][0] if result['recommendations'] else 'N/A'
                    })
                else:
                    display_results.append({
                        'Student': students_data[i].get('name', f'Student {i+1}'),
                        'Predicted Score': 'Error',
                        'Confidence': 'N/A',
                        'Category': 'N/A',
                        'Top Recommendation': result['error']
                    })
            
            results_df = pd.DataFrame(display_results)
            st.success(f"Batch prediction completed for {num_students} students!")
            st.dataframe(results_df, use_container_width=True)

def show_analysis():
    """Show data analysis and visualization"""
    st.header("📊 Data Analysis & Visualization")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Data Overview", "Model Performance", "Feature Analysis", "Feedback Insights"]
    )
    
    if analysis_type == "Data Overview":
        st.subheader("Data Distribution Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic statistics
            data_stats = st.session_state.data_processor.get_data_statistics(st.session_state.data)
            st.write("**Dataset Statistics**")
            stats_display = {
                "Total Students": data_stats.get('total_students', 0),
                "Number of Features": data_stats.get('total_features', 0),
                "Average Final Score": f"{data_stats.get('final_score', {}).get('mean', 0):.1f}",
                "Score Standard Deviation": f"{data_stats.get('final_score', {}).get('std', 0):.1f}"
            }
            
            for key, value in stats_display.items():
                st.write(f"- **{key}:** {value}")
        
        with col2:
            # Correlation with final score
            st.write("**Correlation with Final Score**")
            correlations = data_stats.get('correlations_with_final_score', {})
            if correlations:
                corr_df = pd.DataFrame(
                    list(correlations.items()),
                    columns=['Feature', 'Correlation']
                ).sort_values('Correlation', ascending=False)
                
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            else:
                st.info("No correlation data available")
        
        # Feature distributions
        st.subheader("Feature Distributions")
        feature_to_plot = st.selectbox(
            "Select feature to visualize",
            ['attendance', 'study_hours', 'previous_score', 'assignment_score', 'participation', 'final_score']
        )
        
        if feature_to_plot in st.session_state.data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Distribution of {feature_to_plot}**")
                hist_values = st.session_state.data[feature_to_plot].values
                st.bar_chart(pd.DataFrame(hist_values, columns=[feature_to_plot]))
            
            with col2:
                st.write(f"**Statistics for {feature_to_plot}**")
                stats = data_stats.get(feature_to_plot, {})
                if stats:
                    stat_display = {
                        "Mean": f"{stats.get('mean', 0):.2f}",
                        "Median": f"{stats.get('median', 0):.2f}",
                        "Standard Deviation": f"{stats.get('std', 0):.2f}",
                        "Minimum": f"{stats.get('min', 0):.2f}",
                        "Maximum": f"{stats.get('max', 0):.2f}"
                    }
                    
                    for key, value in stat_display.items():
                        st.write(f"- **{key}:** {value}")
    
    elif analysis_type == "Model Performance":
        st.subheader("Model Performance Analysis")
        
        try:
            model_info = st.session_state.predictor.get_model_info()
            metrics = model_info.get('performance_metrics', {})
            
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                with col2:
                    st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                with col3:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                with col4:
                    st.metric("Explained Variance", f"{metrics.get('explained_variance', 0):.3f}")
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_importance = model_info.get('feature_importance', {})
                if feature_importance:
                    fi_df = pd.DataFrame(
                        list(feature_importance.items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    st.dataframe(fi_df, use_container_width=True, hide_index=True)
                    
                    # Bar chart
                    st.bar_chart(fi_df.set_index('Feature'))
                else:
                    st.info("No feature importance data available")
            
            else:
                st.warning("No performance metrics available for the current model")
        
        except Exception as e:
            st.error(f"Error loading model performance data: {e}")
    
    elif analysis_type == "Feature Analysis":
        st.subheader("Feature Relationship Analysis")
        
        # Feature correlation
        st.write("**Feature Correlation Matrix**")
        numeric_data = st.session_state.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            correlation_matrix = numeric_data.corr()
            st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
        else:
            st.info("No numeric data available for correlation analysis")
    
    elif analysis_type == "Feedback Insights":
        st.subheader("Feedback Analysis")
        
        feedback_stats = st.session_state.feedback_system.get_feedback_statistics()
        
        if feedback_stats['total_feedback'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Feedback Overview**")
                st.metric("Total Feedback", feedback_stats['total_feedback'])
                st.metric("Feedback with Actual Scores", feedback_stats.get('feedback_with_actual_scores', 0))
                if 'mean_absolute_error' in feedback_stats:
                    st.metric("Mean Absolute Error", f"{feedback_stats['mean_absolute_error']:.2f}")
                if 'accuracy_within_5_points' in feedback_stats:
                    st.metric("Accuracy within 5 points", f"{feedback_stats['accuracy_within_5_points']:.1%}")
            
            with col2:
                st.write("**Feedback by User Type**")
                user_types = feedback_stats.get('feedback_by_user_type', {})
                if user_types:
                    for user_type, count in user_types.items():
                        st.write(f"- **{user_type.title()}:** {count}")
                else:
                    st.info("No user type data available")
            
            # Feedback patterns
            patterns = st.session_state.feedback_system.analyze_feedback_patterns()
            if patterns.get('common_issues'):
                st.write("**Common Feedback Patterns**")
                issues = patterns['common_issues']
                for issue, count in issues.items():
                    st.write(f"- **{issue.replace('_', ' ').title()}:** {count} mentions")
        
        else:
            st.info("No feedback data available yet. Submit some predictions with feedback to see insights here.")

def show_feedback():
    """Show feedback management interface"""
    st.header("🔄 Feedback System")
    
    feedback_stats = st.session_state.feedback_system.get_feedback_statistics()
    
    # Feedback overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Feedback", feedback_stats.get('total_feedback', 0))
    
    with col2:
        st.metric("With Actual Scores", feedback_stats.get('feedback_with_actual_scores', 0))
    
    with col3:
        should_retrain, reason = st.session_state.feedback_system.should_retrain_model()
        status_color = "🟢" if not should_retrain else "🟡"
        st.metric("Retraining Status", f"{status_color} {'Not Needed' if not should_retrain else 'Recommended'}")
    
    # Retraining section
    st.subheader("Model Retraining")
    
    if should_retrain:
        st.warning(f"Retraining recommended: {reason}")
        
        if st.button("Retrain Model with Feedback", type="primary"):
            with st.spinner("Retraining model with latest feedback..."):
                result = st.session_state.feedback_system.retrain_model_with_feedback()
            
            if result['retrained']:
                st.success("Model retrained successfully!")
                
                # Show improvement
                col1, col2 = st.columns(2)
                with col1:
                    old_r2 = result['old_metrics'].get('r2', 0)
                    st.metric("Old R² Score", f"{old_r2:.3f}")
                with col2:
                    new_r2 = result['new_metrics'].get('r2', 0)
                    st.metric("New R² Score", f"{new_r2:.3f}", delta=f"{new_r2 - old_r2:+.3f}")
                
                # Reload predictor
                st.session_state.predictor.load_model()
            else:
                st.error(f"Retraining failed: {result['reason']}")
    else:
        st.success("Model performance is satisfactory. No retraining needed at this time.")
    
    # Feedback details
    st.subheader("Feedback Details")
    
    if feedback_stats['total_feedback'] > 0:
        # Load recent feedback
        feedback_df = st.session_state.feedback_system.load_feedback()
        
        # Show recent feedback
        st.write("**Recent Feedback**")
        recent_feedback = feedback_df.tail(10)
        
        # Simplify display
        display_cols = ['timestamp', 'user_type', 'student_id', 'predicted_score', 'actual_score', 'feedback_rating']
        available_cols = [col for col in display_cols if col in recent_feedback.columns]
        
        if available_cols:
            st.dataframe(recent_feedback[available_cols], use_container_width=True)
        else:
            st.info("No feedback data available for display")
        
        # Feedback analysis
        st.subheader("Feedback Analysis")
        analysis = st.session_state.feedback_system.analyze_feedback_patterns()
        
        if analysis.get('common_issues'):
            st.write("**Common Issues Mentioned**")
            for issue, count in analysis['common_issues'].items():
                if count > 0:
                    st.write(f"- **{issue.replace('_', ' ').title()}:** {count} occurrences")
    
    else:
        st.info("""
        No feedback collected yet. 
        
        To start collecting feedback:
        1. Go to the **Prediction** page
        2. Make some predictions
        3. Provide feedback on the accuracy
        4. The system will use this feedback to improve over time
        """)

def show_system():
    """Show system configuration and management"""
    st.header("⚙️ System Configuration")
    
    # System information
    st.subheader("System Information")
    
    try:
        system_status = {
            "Data": {
                "Raw Data": "✅ Available" if os.path.exists("data/raw/student_data.csv") else "❌ Missing",
                "Processed Data": "✅ Available" if os.path.exists("data/processed/cleaned_student_data.csv") else "❌ Missing",
                "Feedback Data": "✅ Available" if os.path.exists("data/feedback/feedback_data.csv") else "❌ Missing"
            },
            "Models": {
                "Trained Model": "✅ Available" if os.path.exists("models/trained_models/best_model.pkl") else "❌ Missing",
                "Model History": "✅ Available" if os.path.exists("models/model_history/model_versions.json") else "❌ Missing"
            },
            "Configuration": {
                "Config File": "✅ Loaded" if st.session_state.app_initialized else "❌ Error"
            }
        }
        
        for category, items in system_status.items():
            st.write(f"**{category}**")
            for item, status in items.items():
                st.write(f"- {item}: {status}")
    
    except Exception as e:
        st.error(f"Error checking system status: {e}")
    
    # Configuration management
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Initialize System", use_container_width=True):
            with st.spinner("Initializing system..."):
                success = initialize_app()
                if success:
                    st.success("System initialized successfully!")
                    st.rerun()
                else:
                    st.error("System initialization failed!")
    
    with col2:
        if st.button("Train New Model", use_container_width=True):
            with st.spinner("Training new model..."):
                result = st.session_state.model_trainer.train_complete_pipeline()
                if 'error' not in result:
                    st.success(f"Model trained successfully! R²: {result['best_metrics']['r2']:.3f}")
                    st.session_state.predictor.load_model()
                    st.rerun()
                else:
                    st.error(f"Model training failed: {result['error']}")
    
    # Data management
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                sample_data = st.session_state.data_processor.generate_sample_data(500)
                st.session_state.data_processor.save_raw_data(sample_data)
                st.session_state.data = st.session_state.data_processor.load_processed_data()
                st.success("Sample data generated successfully!")
                st.rerun()
    
    with col2:
        if st.button("Clear Feedback Data", use_container_width=True):
            if st.checkbox("I understand this will delete all feedback data"):
                try:
                    os.remove("data/feedback/feedback_data.csv")
                    st.success("Feedback data cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing feedback data: {e}")
    
    # System reports
    st.subheader("System Reports")
    
    if st.button("Generate Comprehensive Report", use_container_width=True):
        with st.spinner("Generating reports..."):
            # This would generate comprehensive reports
            st.info("""
            **Report Generation Complete**
            
            The following reports would be generated:
            - Model performance analysis
            - Data quality assessment
            - Feedback impact analysis
            - System health check
            
            In a full implementation, these would be saved as PDF files.
            """)

if __name__ == "__main__":
    main()