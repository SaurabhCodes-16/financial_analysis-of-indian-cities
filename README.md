# Financial Analysis of Indian Cities

## Overview
This project performs comprehensive data mining and machine learning analysis on financial data from Indian cities. The analysis includes exploratory data analysis, clustering, classification, model explainability using SHAP, and disparity analysis across different city tiers.

## Project Structure
```
├── data/
│   └── finance.csv                    # Raw financial dataset
├── outputs/
│   ├── phase1_processed.csv           # Processed data after EDA
│   ├── phase2_clustered.csv           # Data with cluster labels
│   ├── phase3_test_predictions.csv    # Classification predictions
│   ├── random_forest_predictions.csv  # Random Forest predictions
│   ├── random_forest_results.txt      # Model evaluation results
│   ├── shap_feature_importance.txt    # SHAP feature importance
│   └── shap_prediction_details.txt    # SHAP prediction explanations
├── plots/                             # Generated visualization plots
├── phase1_eda_feature_engineering.py  # Exploratory Data Analysis & Feature Engineering
├── phase2_clustering.py               # K-Means Clustering Analysis
├── phase2_visualization.py            # Clustering Visualization
├── phase3_classification.py           # XGBoost Classification Model
├── phase4_shap_analysis.py            # SHAP Explainability Analysis
├── phase5_disparity_analysis.py       # Disparity Analysis Across City Tiers
├── random_forest_model.py             # Random Forest Classification Model
├── LICENSE                            # Project license
└── README.md                          # This file
```

## Phases

### Phase 1: Exploratory Data Analysis & Feature Engineering
- Loads and examines the financial dataset
- Handles missing values using median imputation
- Performs feature engineering and data preprocessing
- Generates initial visualizations

### Phase 2: Clustering Analysis
- Applies K-Means clustering on standardized features
- Identifies distinct city groups based on financial characteristics
- Creates cluster visualizations

### Phase 3: Classification
- Trains XGBoost classifier to predict city vulnerability levels
- Evaluates model performance on test data
- Saves trained model for later use

### Phase 4: SHAP Analysis
- Uses SHAP (SHapley Additive exPlanations) to explain model predictions
- Analyzes feature importance and individual prediction explanations
- Generates SHAP summary plots

### Phase 5: Disparity Analysis
- Analyzes financial disparities across different city tiers
- Performs statistical tests (ANOVA) to identify significant differences
- Examines expense ratios across city categories

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- shap
- joblib
- scipy

## Installation
1. Clone or download the repository
2. Install required Python packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib scipy
   ```

## Usage
Run the phases sequentially:

1. **Phase 1**: `python phase1_eda_feature_engineering.py`
2. **Phase 2**: `python phase2_clustering.py` then `python phase2_visualization.py`
3. **Phase 3**: `python phase3_classification.py`
4. **Phase 4**: `python phase4_shap_analysis.py`
5. **Phase 5**: `python phase5_disparity_analysis.py`

Alternatively, run the Random Forest model: `python random_forest_model.py`

## Dataset
The analysis uses `data/finance.csv` containing financial metrics for Indian cities including:
- Income levels
- Expense ratios
- Savings gaps
- Recovery rates
- Demographic data (dependents, age)
- City tier classifications

## Outputs
- Processed datasets in `outputs/`
- Visualization plots in `plots/`
- Model evaluation results and SHAP analyses in text files

## License
See LICENSE file for details.