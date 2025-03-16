import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# Function to get user input
def get_user_input():
    print("\nHEART DISEASE RISK ASSESSMENT")
    print("=============================")
    print("Please enter the following details:")
    
    sex = int(input("Enter sex (0 for Female, 1 for Male): "))
    
    age_mapping = {
        0: "18-24", 1: "25-29", 2: "30-34", 3: "35-39", 
        4: "40-44", 5: "45-49", 6: "50-54", 7: "55-59",
        8: "60-64", 9: "65-69", 10: "70-74", 11: "75-79", 12: "80+"
    }
    print("\nAge Categories:")
    for key, value in age_mapping.items():
        print(f"{key}: {value}")
    age_category = int(input("Enter age category number (0-12): "))
    
    weight = float(input("Enter weight in kilograms: "))
    bmi = float(input("Enter BMI: "))
    
    # Health conditions (binary inputs)
    print("\nFor the following health conditions, enter 1 for Yes or 0 for No:")
    had_angina = int(input("Have you had angina or coronary heart disease? "))
    had_stroke = int(input("Have you had a stroke? "))
    had_copd = int(input("Have you had COPD, emphysema or chronic bronchitis? "))
    had_kidney_disease = int(input("Have you had kidney disease? "))
    had_arthritis = int(input("Have you had arthritis? "))
    had_diabetes = int(input("Have you had diabetes? "))
    
    # Difficulties (binary inputs)
    deaf_hard_hearing = int(input("Are you deaf or have serious difficulty hearing? "))
    blind_vision_difficulty = int(input("Are you blind or have serious difficulty seeing? "))
    difficulty_walking = int(input("Do you have serious difficulty walking or climbing stairs? "))
    difficulty_dressing_bathing = int(input("Do you have difficulty dressing or bathing? "))
    difficulty_errands = int(input("Do you have difficulty doing errands alone? "))
    
    # Smoking status
    smoker_mapping = {0: "Never smoked", 1: "Former smoker", 
                      2: "Current smoker - some days", 3: "Current smoker - every day"}
    print("\nSmoker Status:")
    for key, value in smoker_mapping.items():
        print(f"{key}: {value}")
    smoker_status = int(input("Enter smoker status (0-3): "))
    
    chest_scan = int(input("Have you had a chest scan in the past year (1 for Yes, 0 for No)? "))
    alcohol_drinkers = int(input("Do you drink alcohol (1 for Yes, 0 for No)? "))
    pneumo_vax_ever = int(input("Have you ever had a pneumonia vaccination (1 for Yes, 0 for No)? "))
    
    # Return input as a dictionary
    return {
        'Sex': [sex],
        'AgeCategory_Ordinal': [age_category],
        'WeightInKilograms': [weight],
        'BMI': [bmi],
        'HadAngina': [had_angina],
        'HadStroke': [had_stroke],
        'HadCOPD': [had_copd],
        'HadKidneyDisease': [had_kidney_disease],
        'HadArthritis': [had_arthritis],
        'HadDiabetes': [had_diabetes],
        'DeafOrHardOfHearing': [deaf_hard_hearing],
        'BlindOrVisionDifficulty': [blind_vision_difficulty],
        'DifficultyWalking': [difficulty_walking],
        'DifficultyDressingBathing': [difficulty_dressing_bathing],
        'DifficultyErrands': [difficulty_errands],
        'SmokerStatus': [smoker_status],
        'ChestScan': [chest_scan],
        'AlcoholDrinkers': [alcohol_drinkers],
        'PneumoVaxEver': [pneumo_vax_ever]
    }

# Main function
def main():
    # Only use the three specified models
    models = ['random_forest', 'gradient_boosting', 'stacking']
    
    # Get user input for prediction
    user_data = get_user_input()
    
    print("\nPROCESSING YOUR HEART DISEASE RISK ASSESSMENT")
    print("============================================")
    
    # Predict for each model
    for model_name in models:
        try:
            # Load the model
            model = load_model(model_name)
            
            # Predict using the user input data
            prediction, risk_percentage = predict_heart_attack(model, user_data)
            
            # Display the result with a clear visual indicator
            print(f"\n{model_name.upper()} MODEL:")
            risk_status = "HIGH RISK" if prediction[0] == 1 else "LOW RISK"
            print(f"Result: {risk_status} OF HEART ATTACK")
            
            if risk_percentage is not None:
                print(f"Risk Percentage: {risk_percentage[0]:.2f}%")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"\nError with {model_name} model: {str(e)}")
    
    print("\nAssessment complete. Consult with a healthcare professional for proper medical advice.")

if __name__ == "__main__":
    main()