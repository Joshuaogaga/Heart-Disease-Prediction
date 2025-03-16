import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    try:
        models['random_forest'] = joblib.load('models/random_forest_model.pkl')
        models['gradient_boosting'] = joblib.load('models/gradient_boosting_model.pkl')
        models['stacking'] = joblib.load('models/stacking_model.pkl')
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Function to predict heart attack risk
def predict_heart_attack(model, input_data):
    # Create DataFrame without column names
    input_df = pd.DataFrame(input_data)
    
    # Convert to numpy array to remove feature names
    input_array = input_df.values
    
    # Making predictions
    predictions = model.predict(input_array)
    
    # Get probability scores if the model supports it
    try:
        probabilities = model.predict_proba(input_array)
        risk_percentage = probabilities[:, 1] * 100
    except:
        risk_percentage = [None] * len(predictions)
    
    return predictions, risk_percentage

# Main app
def main():
    # Header
    st.title("❤️ Heart Disease Risk Predictor")
    st.write("Enter your health information below to assess your risk of heart disease.")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("Failed to load models. Please check if model files exist in the 'models' directory.")
        return
    
    # Create sidebar for inputs
    st.sidebar.title("Your Health Information")
    
    # Demographics
    st.sidebar.header("Demographics")
    sex = st.sidebar.radio("Sex", options=["Female", "Male"])
    sex_value = 0 if sex == "Female" else 1
    
    age_mapping = {
        "18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, 
        "40-44": 4, "45-49": 5, "50-54": 6, "55-59": 7,
        "60-64": 8, "65-69": 9, "70-74": 10, "75-79": 11, "80+": 12
    }
    age_category = st.sidebar.selectbox("Age Group", options=list(age_mapping.keys()))
    age_category_value = age_mapping[age_category]
    
    # Physical measurements
    st.sidebar.header("Physical Measurements")
    weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    bmi = st.sidebar.number_input("BMI", min_value=15.0, max_value=50.0, value=24.5, step=0.1)
    
    # Health conditions
    st.sidebar.header("Health Conditions")
    had_angina = st.sidebar.radio("Angina or coronary heart disease", options=["No", "Yes"]) == "Yes"
    had_stroke = st.sidebar.radio("Stroke", options=["No", "Yes"]) == "Yes"
    had_copd = st.sidebar.radio("COPD, emphysema or chronic bronchitis", options=["No", "Yes"]) == "Yes"
    had_kidney_disease = st.sidebar.radio("Kidney disease", options=["No", "Yes"]) == "Yes"
    had_arthritis = st.sidebar.radio("Arthritis", options=["No", "Yes"]) == "Yes"
    had_diabetes = st.sidebar.radio("Diabetes", options=["No", "Yes"]) == "Yes"
    
    # Physical difficulties
    st.sidebar.header("Physical Difficulties")
    deaf_hard_hearing = st.sidebar.radio("Deaf or serious difficulty hearing", options=["No", "Yes"]) == "Yes"
    blind_vision_difficulty = st.sidebar.radio("Blind or serious difficulty seeing", options=["No", "Yes"]) == "Yes"
    difficulty_walking = st.sidebar.radio("Serious difficulty walking/climbing stairs", options=["No", "Yes"]) == "Yes"
    difficulty_dressing_bathing = st.sidebar.radio("Difficulty dressing or bathing", options=["No", "Yes"]) == "Yes"
    difficulty_errands = st.sidebar.radio("Difficulty doing errands alone", options=["No", "Yes"]) == "Yes"
    
    # Lifestyle factors
    st.sidebar.header("Lifestyle Factors")
    smoker_options = {
        "Never smoked": 0, 
        "Former smoker": 1, 
        "Current smoker - some days": 2, 
        "Current smoker - every day": 3
    }
    smoker_status = st.sidebar.selectbox("Smoking Status", options=list(smoker_options.keys()))
    smoker_status_value = smoker_options[smoker_status]
    
    chest_scan = st.sidebar.radio("Had a chest scan in the past year", options=["No", "Yes"]) == "Yes"
    alcohol_drinkers = st.sidebar.radio("Consume alcohol", options=["No", "Yes"]) == "Yes"
    pneumo_vax_ever = st.sidebar.radio("Ever had a pneumonia vaccination", options=["No", "Yes"]) == "Yes"
    
    # Prepare input data
    input_data = {
        'Sex': [sex_value],
        'AgeCategory_Ordinal': [age_category_value],
        'WeightInKilograms': [weight],
        'BMI': [bmi],
        'HadAngina': [1 if had_angina else 0],
        'HadStroke': [1 if had_stroke else 0],
        'HadCOPD': [1 if had_copd else 0],
        'HadKidneyDisease': [1 if had_kidney_disease else 0],
        'HadArthritis': [1 if had_arthritis else 0],
        'HadDiabetes': [1 if had_diabetes else 0],
        'DeafOrHardOfHearing': [1 if deaf_hard_hearing else 0],
        'BlindOrVisionDifficulty': [1 if blind_vision_difficulty else 0],
        'DifficultyWalking': [1 if difficulty_walking else 0],
        'DifficultyDressingBathing': [1 if difficulty_dressing_bathing else 0],
        'DifficultyErrands': [1 if difficulty_errands else 0],
        'SmokerStatus': [smoker_status_value],
        'ChestScan': [1 if chest_scan else 0],
        'AlcoholDrinkers': [1 if alcohol_drinkers else 0],
        'PneumoVaxEver': [1 if pneumo_vax_ever else 0]
    }
    
    # Count positive health conditions for context
    health_conditions = [
        'HadAngina', 'HadStroke', 'HadCOPD', 'HadKidneyDisease', 
        'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing',
        'BlindOrVisionDifficulty', 'DifficultyWalking', 
        'DifficultyDressingBathing', 'DifficultyErrands'
    ]
    positive_conditions = sum(input_data[col][0] for col in health_conditions)
    
    # Predict button
    predict_button = st.sidebar.button("Predict Heart Disease Risk", type="primary")
    
    # Main content area
    if predict_button:
        st.header("Heart Disease Risk Assessment Results")
        
        # Make predictions with all models
        results = {}
        
        # Debug information
        st.subheader("Debug Information")
        debug_expander = st.expander("View Input Data and Model Diagnostics")
        with debug_expander:
            st.write("Input Data:")
            st.write(pd.DataFrame(input_data))
            st.write(f"Number of positive health conditions: {positive_conditions}")
            
            # Check if our models have feature names we can access
            for model_name, model in models.items():
                st.write(f"Model: {model_name}")
                if hasattr(model, "feature_names_in_"):
                    st.write(f"Expected features: {model.feature_names_in_.tolist()}")
                if hasattr(model, "classes_"):
                    st.write(f"Classes: {model.classes_}")
                if hasattr(model, "feature_importances_"):
                    importance_df = pd.DataFrame({
                        'Feature': list(input_data.keys()),
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    st.write("Feature Importances:")
                    st.dataframe(importance_df)
        
        for model_name, model in models.items():
            prediction, risk_percentage = predict_heart_attack(model, input_data)
            results[model_name] = {
                'prediction': prediction[0],
                'risk_percentage': risk_percentage[0]
            }
            
            # Add debug info to expander
            with debug_expander:
                st.write(f"{model_name} prediction: {prediction[0]}")
                st.write(f"{model_name} risk percentage: {risk_percentage[0]:.2f}%")
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Random Forest Model")
            # SWAP HIGH/LOW RISK INTERPRETATION: 0 is high risk, 1 is low risk
            rf_result = "HIGH RISK" if results['random_forest']['prediction'] == 0 else "LOW RISK"
            rf_color = "red" if results['random_forest']['prediction'] == 0 else "green"
            st.markdown(f"**Result:** <span style='color:{rf_color};font-size:20px'>{rf_result}</span>", unsafe_allow_html=True)
            st.markdown(f"**Risk Percentage:** {results['random_forest']['risk_percentage']:.2f}%")
            
            # Simple progress bar for risk visualization
            st.progress(results['random_forest']['risk_percentage'] / 100)
            
            # Add explanation based on health conditions
            if positive_conditions == 0 and rf_result == "LOW RISK":
                st.info("Low risk prediction based on lack of health conditions")
            elif positive_conditions > 0 and rf_result == "HIGH RISK":
                st.warning(f"High risk prediction based on {positive_conditions} health conditions")
        
        with col2:
            st.subheader("Gradient Boosting Model")
            # SWAP HIGH/LOW RISK INTERPRETATION: 0 is high risk, 1 is low risk
            gb_result = "HIGH RISK" if results['gradient_boosting']['prediction'] == 0 else "LOW RISK"
            gb_color = "red" if results['gradient_boosting']['prediction'] == 0 else "green"
            st.markdown(f"**Result:** <span style='color:{gb_color};font-size:20px'>{gb_result}</span>", unsafe_allow_html=True)
            st.markdown(f"**Risk Percentage:** {results['gradient_boosting']['risk_percentage']:.2f}%")
            
            # Simple progress bar for risk visualization
            st.progress(results['gradient_boosting']['risk_percentage'] / 100)
            
            # Add explanation based on health conditions
            if positive_conditions == 0 and gb_result == "LOW RISK":
                st.info("Low risk prediction based on lack of health conditions")
            elif positive_conditions > 0 and gb_result == "HIGH RISK":
                st.warning(f"High risk prediction based on {positive_conditions} health conditions")
            
        with col3:
            st.subheader("Stacking Ensemble Model")
            # SWAP HIGH/LOW RISK INTERPRETATION: 0 is high risk, 1 is low risk
            stack_result = "HIGH RISK" if results['stacking']['prediction'] == 0 else "LOW RISK"
            stack_color = "red" if results['stacking']['prediction'] == 0 else "green"
            st.markdown(f"**Result:** <span style='color:{stack_color};font-size:20px'>{stack_result}</span>", unsafe_allow_html=True)
            st.markdown(f"**Risk Percentage:** {results['stacking']['risk_percentage']:.2f}%")
            
            # Simple progress bar for risk visualization
            st.progress(results['stacking']['risk_percentage'] / 100)
            
            # Add explanation based on health conditions
            if positive_conditions == 0 and stack_result == "LOW RISK":
                st.info("Low risk prediction based on lack of health conditions")
            elif positive_conditions > 0 and stack_result == "HIGH RISK":
                st.warning(f"High risk prediction based on {positive_conditions} health conditions")
        
        # Model comparison chart
        st.header("Model Comparison")
        
        # Create bar chart data
        chart_data = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting', 'Stacking Ensemble'],
            'Risk Percentage': [
                results['random_forest']['risk_percentage'],
                results['gradient_boosting']['risk_percentage'],
                results['stacking']['risk_percentage']
            ]
        })
        
        # Simple bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            chart_data['Model'], 
            chart_data['Risk Percentage'],
            color=['#1E88E5', '#FFC107', '#4CAF50']
        )
        
        # Add risk percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{height:.2f}%',
                ha='center',
                va='bottom'
            )
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('Risk Percentage (%)')
        ax.set_title('Heart Disease Risk Prediction Comparison')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Recommendations section
        st.header("Recommendations")
        st.write("Based on your information, here are some general recommendations:")
        st.write("""
        - Regular check-ups with your healthcare provider
        - Maintain a healthy diet rich in fruits, vegetables, and whole grains
        - Regular physical activity (aim for at least 150 minutes per week)
        - Maintain a healthy weight
        - Limit alcohol consumption
        - Quit smoking if applicable
        - Manage stress effectively
        """)
        
        st.caption("Disclaimer: This tool provides an estimate based on machine learning models and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.")
    
    else:
        # Show information about the app when not predicting
        st.write("This application uses three machine learning models to predict your risk of heart disease based on your health information.")
        st.write("Fill in your information in the sidebar and click 'Predict Heart Disease Risk' to see your results.")
        
        st.header("About the Models")
        st.write("""
        - **Random Forest**: A robust model that combines multiple decision trees for accurate predictions.
        - **Gradient Boosting**: An advanced algorithm that builds trees sequentially to improve prediction accuracy.
        - **Stacking Ensemble**: A powerful meta-model that combines predictions from multiple models for better results.
        """)
        
        st.info("Heart disease risk is influenced by many factors including age, lifestyle, and existing health conditions. This tool helps assess your personal risk based on these factors.")

if __name__ == "__main__":
    main()