import os
# Set a specific number of cores
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier

# Function to preprocess data
def preprocess_data(df):
    # Separate features (X) and target (y)
    X = df.drop('HadHeartAttack', axis=1)
    y = df['HadHeartAttack']
    
    # Balance data using SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    
    print("Class distribution after SMOTE:")
    print(y.value_counts())
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Define models to train
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest_Neighbors': KNeighborsClassifier(n_neighbors=3, metric='euclidean'),
        'Naive_Bayes': GaussianNB(),
        'Random_Forest': RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42),
        'Gradient_Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n===== {model_name} =====")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store the trained model in memory (but don't save to disk)
        trained_models[model_name] = model
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100
        
        # Store results
        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
        
        # Print metrics
        print(f'Accuracy Score: {accuracy:.2f}%')
        print(f'Precision: {precision:.2f}%')
        print(f'Recall: {recall:.2f}%')
        print(f'F1 Score: {f1:.2f}%')
        
        # Print classification report
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))
        
        print('=' * 50)
    
    return results, trained_models

# Create stacking ensemble model
def create_stacking_model(X_train, X_test, y_train, y_test, trained_models):
    print("\n===== Creating Stacking Ensemble =====")
    
    # Get the best models
    rf_model = trained_models['Random_Forest']
    knn_model = trained_models['K-Nearest_Neighbors']
    gb_model = trained_models['Gradient_Boosting']
    
    # Define base estimators
    estimators = [
        ('rf', rf_model),
        ('knn', knn_model),
        ('gb', gb_model)
    ]
    
    # Create stacking classifier
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method='predict_proba'
    )
    
    # Train the stacking model
    stack_model.fit(X_train, y_train)
    
    # Evaluate the stacking model
    y_pred_stack = stack_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_stack) * 100
    precision = precision_score(y_test, y_pred_stack) * 100
    recall = recall_score(y_test, y_pred_stack) * 100
    f1 = f1_score(y_test, y_pred_stack) * 100
    
    # Create results dictionary
    stack_results = {
        'Stacking_Ensemble': {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
    }
    
    # Print results
    print(f"Accuracy Score: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred_stack))
    
    print('=' * 50)
    
    return stack_results

# Main function
def main():
    print("Loading and preprocessing data...")
    
    # Load the preprocessed data
    df = pd.read_csv('C:/Users/joshu/Documents/Winter 2025/Heart-Disease-Prediction/data/processed.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train and evaluate individual models
    results, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Create and evaluate stacking model
    stack_results = create_stacking_model(X_train, X_test, y_train, y_test, trained_models)
    
    # Combine all results
    all_results = {**results, **stack_results}
    
    # Find and print the best model
    best_model = max(all_results, key=lambda x: all_results[x]['F1_Score'])
    print(f"\nBest Model: {best_model}")
    print(f"F1 Score: {all_results[best_model]['F1_Score']:.2f}%")
    
    # Save results to CSV
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df.to_csv('model_results.csv')
    print("Results saved to 'model_results.csv'")

if __name__ == '__main__':
    main()