# Healthcare Stroke Risk Prediction

An end-to-end healthcare analytics project that predicts a patient's **stroke risk** from clinical and demographic data, and deploys the trained model as an interactive Streamlit app. Built as a 4-milestone project: data preprocessing → analysis/visualization → model development → deployment.

## Project Overview

- **Task:** Binary classification — predict `stroke` (0/1) from patient health and demographic features
- **Dataset:** Healthcare stroke dataset (age, hypertension, heart disease, marital status, work type, residence type, glucose level, BMI, smoking status)
- **Key challenge:** Severe class imbalance (~5% positive stroke cases), handled with SMOTE and class weighting
- **Outcome:** Random Forest selected as the best-performing model, deployed via Streamlit

## Milestones

### Milestone 1 — Data Collection, Exploration & Preprocessing
- Loaded the raw dataset (5,110 records, 13 columns), checked shape/dtypes/duplicates
- Explored missing values and outliers (age, glucose level, BMI) via box plots and a missing-values heatmap
- Filled missing `bmi` with the median, standardized `smoking_status` labels
- One-hot encoded categorical features (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) and exported `preprocessed_stroke_data.csv`

### Milestone 2 — Data Analysis & Visualization
- Distribution plots for numeric features (age, glucose level, BMI) and count plots for categorical features
- Statistical tests: t-tests for age and glucose level vs. stroke (both significant, p < 0.001), chi-square test for gender vs. stroke (not significant)
- Logistic regression (statsmodels) for an initial look at feature relationships with stroke
- Correlation heatmap, bivariate age-vs-stroke boxplot, and an interactive Dash/Plotly exploration (bar, pie, scatter)

### Milestone 3 — Predictive Model Development & Optimization
- Trained and compared four models: **Logistic Regression, Random Forest, Gradient Boosting, and a Neural Network**, using SMOTE-resampled training data and class weighting to address imbalance
- Evaluated each with accuracy, precision, recall, F1, and ROC AUC — with confusion matrices per model
- Tuned Random Forest via `RandomizedSearchCV` then `GridSearchCV`
- **Random Forest** selected as the final model and serialized (`stroke_model.pkl`, `scaler.pkl`, `model_columns.pkl`) for deployment

### Milestone 4 — Deployment
- Streamlit web app (`app/app.py`) that takes patient details as input and returns a predicted stroke probability with a risk flag (threshold: 0.3)
- Full write-up and results in the accompanying presentation (`presentation/Health_Care_Presentation.pptx`)

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.918 | 0.328 | 0.323 | 0.325 | 0.846 |
| Random Forest | 0.920 | 0.167 | 0.081 | 0.109 | 0.527 |
| Gradient Boosting | 0.878 | 0.138 | 0.194 | 0.161 | 0.558 |
| Neural Network | 0.890 | 0.195 | 0.258 | 0.222 | 0.595 |

> On a highly imbalanced dataset like this (~5% positive class), accuracy alone is misleading — a model predicting "no stroke" for everyone would already score ~95%. Logistic Regression has the strongest ROC AUC and the most balanced precision/recall here, which is worth highlighting alongside Random Forest (the model that was ultimately deployed) if this project comes up in an interview.

## Tech Stack

- Python 3, pandas, numpy
- matplotlib, seaborn, plotly, dash — visualization
- scikit-learn — LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, model selection & tuning
- imbalanced-learn (SMOTE) — class imbalance handling
- statsmodels, scipy — statistical testing
- Streamlit — model deployment
- pickle — model/scaler/column serialization

## Repository Structure

```
healthcare-stroke-prediction/
├── notebooks/
│   ├── Milestone_1_Data_Collection_Preprocessing.ipynb
│   ├── Milestone_2_Analysis_Visualization.ipynb
│   └── Milestone_3_Model_Development_Optimization.ipynb
├── app/
│   └── app.py                  # Streamlit stroke risk prediction app
├── models/                     # stroke_model.pkl, scaler.pkl, model_columns.pkl (generated, not committed)
├── presentation/
│   └── Health_Care_Presentation.pptx
├── data/                       
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/healthcare-stroke-prediction.git
cd healthcare-stroke-prediction
pip install -r requirements.txt
```

1. Place the raw dataset at `data/healthcare_stroke_dataset.csv`
2. Run the notebooks in order (Milestone 1 → 2 → 3) — Milestone 3 saves `stroke_model.pkl`, `scaler.pkl`, and `model_columns.pkl` into `models/`
3. Launch the app:
```bash
streamlit run app/app.py
```
## Deployment Demo
https://youtu.be/VglJZdsdtq8
## Author

Sohila — Data Science graduate, Sadat Academy for Management Sciences
