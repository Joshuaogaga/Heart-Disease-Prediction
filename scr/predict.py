import joblib
import pandas as pd
import numpy as np
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Function to load the trained model
def load_model(model_name):
    # Load the model from the specified path
    model_path = f'models/{model_name}_model.pkl'
    model = joblib.load(model_path)
    return model

# Function to predict heart attack risk using a model
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

# Main function
def main():
    # Only use the three specified models
    models = ['random_forest', 'gradient_boosting', 'stacking']
    
    # Sample input data (replace with actual values for prediction)
    sample_data = {
        'Sex': [1],  # Male
        'AgeCategory_Ordinal': [8],  # Age 60-64
        'WeightInKilograms': [85.5],
        'BMI': [28.4],
        'HadAngina': [0],
        'HadStroke': [0],
        'HadCOPD': [0],
        'HadKidneyDisease': [0],
        'HadArthritis': [1],
        'HadDiabetes': [1],
        'DeafOrHardOfHearing': [0],
        'BlindOrVisionDifficulty': [0],
        'DifficultyWalking': [1],
        'DifficultyDressingBathing': [0],
        'DifficultyErrands': [0],
        'SmokerStatus': [2],  # Former smoker
        'ChestScan': [0],
        'AlcoholDrinkers': [1],
        'PneumoVaxEver': [1]
    }
    
    print("\nHEART DISEASE RISK PREDICTION")
    print("===========================")
    
    # Predict for each model
    for model_name in models:
        try:
            # Load the model
            model = load_model(model_name)
            
            # Predict using the sample data
            prediction, risk_percentage = predict_heart_attack(model, sample_data)
            
            print(f"\n{model_name.upper()} MODEL:")
            risk_status = "HIGH RISK" if prediction[0] == 1 else "LOW RISK"
            print(f"Result: {risk_status} OF HEART ATTACK")
            
            if risk_percentage[0] is not None:
                print(f"Risk Percentage: {risk_percentage[0]:.2f}%")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"\nError with {model_name} model: {str(e)}")
    
    print("\nPrediction complete. This is for demonstration purposes only.")
    print("Always consult with a healthcare professional for proper medical advice.")

if __name__ == "__main__":
    main()