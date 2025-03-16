# Heart Disease Prediction Project

## Group Members
- Onajemo Joshua
- Precious Nwaokenneya

---

## Project Overview

### Project Title
Heart Disease Prediction

### Introduction
Heart disease is one of the leading causes of mortality worldwide, with heart attacks being a primary manifestation. Early detection and intervention can significantly reduce the risk and improve patient outcomes. This project aims to predict whether a patient has had a heart attack based on various demographic, health, and lifestyle features. We leverage machine learning models to enhance prediction accuracy and provide actionable insights for healthcare providers.

---

## Dataset
The dataset contains medical and lifestyle information from patients, focusing on features such as age, BMI, smoking status, medical history, and more. Each record represents an individual's health data related to heart disease.

### Key Features
- **Demographic Information**: Age, Sex, Race/Ethnicity, State of residence.
- **Health Conditions**: General health status, BMI, history of diseases (e.g., diabetes, asthma, stroke, kidney disease, etc.).
- **Lifestyle Factors**: Smoking status, alcohol consumption, physical activity, vaccination history, etc.
- **Target Variable**: `HadHeartAttack` (1 for Yes, 0 for No).

---

## Technologies Used
- **Python**: Primary programming language.
- **Pandas and NumPy**: For data manipulation and preprocessing.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Matplotlib and Seaborn**: For data visualization.
- **Joblib**: For saving and loading trained models.
- **Streamlit** (optional): For deploying the model as a web application.

---

## How to Access and Run the Project

### Prerequisites
- Python 3.x (recommended: 3.8 or higher).
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `streamlit` (optional).

### Steps to Access the Project
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Joshuaogaga/Heart-Disease-Prediction.git
   cd heart-disease-prediction
   ```

2. **Install Required Libraries**
   ```bash
   conda env create -f genv.yaml
   ```
   ```bash
   conda activate genv
   ```

3. **Run the Project**
   - To preprocess the data and train the model, navigate to the `scripts/` folder and run:
     ```bash
     python scr/train.py
     ```
   - To evaluate the model and make predictions, run:
     ```bash
     python scr/predict.py
     ```
   - If you prefer inputting your own data for prediction, run:
     ```bash
     python scr/predict_input.py
     ```

4. **Deploy the Model (Optional)**
   - If you want to deploy the model as a web application using Streamlit, run:
     ```bash
     streamlit run scr/app.py
     ```
   - **Please note**, random_forest_model.pkl and stacking_model.pkl was not uploaded here due to his large size. So to successfully run this project, kindly save this models on the model folder before running it to avoid errors.
---

## Evaluation Metrics
- **Accuracy**: Overall correct predictions.
- **Precision and Recall**: To handle class imbalance.
- **F1 Score**: Balance between precision and recall.
- **ROC-AUC Score**: To evaluate the model's performance across different thresholds.

---

## Repository Structure
```
heart-disease-prediction/
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks for EDA and modeling
├── scr/                    # Python scripts for data preprocessing and model training
├── models/                 # Saved models 
├── README.md               # Project description and instructions
└── requirements.txt        # Dependencies
```

---

## References
- Dataset source: [(https://www.kaggle.com/datasets/tarekmuhammed/patients-data-for-medical-field/data)]
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

