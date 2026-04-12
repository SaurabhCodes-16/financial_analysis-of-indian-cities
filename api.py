from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import traceback
import shap

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

model = joblib.load('outputs/random_forest_model.pkl')
scaler = joblib.load('outputs/random_forest_scaler.pkl')

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

FEATURE_COLUMNS = [
    'Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 
    'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 
    'Miscellaneous', 'Desired_Savings_Percentage', 'Desired_Savings', 'Disposable_Income', 
    'Potential_Savings_Groceries', 'Potential_Savings_Transport', 'Potential_Savings_Eating_Out', 
    'Potential_Savings_Entertainment', 'Potential_Savings_Utilities', 'Potential_Savings_Healthcare', 
    'Potential_Savings_Education', 'Potential_Savings_Miscellaneous', 'Total_Expenses', 
    'Total_Potential_Savings', 'Savings_Gap', 'Expense_Ratio', 'Recovery_Rate', 
    'City_Tier_Tier_2', 'City_Tier_Tier_3', 'Occupation_Retired', 'Occupation_Self_Employed', 
    'Occupation_Student'
]

def safe_float(val):
    try:
        if val is None or str(val).strip() == '':
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def preprocess_input(data):
    # Initialize basic features with defaults to 0 if missing
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    
    # Simple mapping
    row['Income'] = safe_float(data.get('Income'))
    row['Age'] = safe_float(data.get('Age'))
    row['Dependents'] = safe_float(data.get('Dependents'))
    
    expenses = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 
                'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']
    total_expenses = 0
    for exp in expenses:
        val = safe_float(data.get(exp))
        row[exp] = val
        total_expenses += val
        
    row['Desired_Savings_Percentage'] = safe_float(data.get('Desired_Savings_Percentage'))
    row['Desired_Savings'] = (row['Income'] * row['Desired_Savings_Percentage']) / 100.0
    row['Disposable_Income'] = row['Income'] - total_expenses
    row['Total_Expenses'] = total_expenses
    
    # Potential Savings (Assume user specifies them, otherwise 0)
    potential_cols = [
        'Potential_Savings_Groceries', 'Potential_Savings_Transport', 'Potential_Savings_Eating_Out',
        'Potential_Savings_Entertainment', 'Potential_Savings_Utilities', 'Potential_Savings_Healthcare',
        'Potential_Savings_Education', 'Potential_Savings_Miscellaneous'
    ]
    total_potential = 0
    for p_col in potential_cols:
        val = safe_float(data.get(p_col))
        row[p_col] = val
        total_potential += val
        
    row['Total_Potential_Savings'] = total_potential
    row['Savings_Gap'] = row['Desired_Savings'] - row['Disposable_Income']
    row['Expense_Ratio'] = row['Total_Expenses'] / (row['Income'] if row['Income'] > 0 else 1)
    row['Recovery_Rate'] = row['Total_Potential_Savings'] / (np.abs(row['Savings_Gap']) + 1)
    
    # Categorical
    city_tier = data.get('City_Tier', 'Tier 1')
    if city_tier == 'Tier 2':
        row['City_Tier_Tier_2'] = 1.0
    elif city_tier == 'Tier 3':
        row['City_Tier_Tier_3'] = 1.0
        
    occupation = data.get('Occupation', 'Employed') # Employed is default (dropped)
    if occupation == 'Retired':
        row['Occupation_Retired'] = 1.0
    elif occupation == 'Self-Employed':
        row['Occupation_Self_Employed'] = 1.0
    elif occupation == 'Student':
        row['Occupation_Student'] = 1.0
        
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return df, row

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df, row = preprocess_input(data)
        
        # Scale features
        scaled_features = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_COLUMNS)
        
        # Predict
        prediction = model.predict(scaled_df)[0]
        pred_int = int(prediction)
        
        # Calculate SHAP values
        shap_values_all = explainer.shap_values(scaled_df)
        # explainer returns array of shape (1, 35, 3) for multi-class with 1 sample
        pred_shap_vals = shap_values_all[0, :, pred_int] # Get SHAP values for the predicted class
        
        shap_list = [{"feature": f, "value": float(v)} for f, v in zip(FEATURE_COLUMNS, pred_shap_vals)]
        # Sort by absolute impact and take top 5
        shap_list = sorted(shap_list, key=lambda x: abs(x["value"]), reverse=True)[:5]
        
        # Mapping back
        label_map_inverse = {
            0: "Stable",
            1: "At Risk",
            2: "Vulnerable"
        }
        status = label_map_inverse.get(pred_int, "Unknown")
        
        return jsonify({
            "status": status,
            "metrics": {
                "Total_Expenses": row['Total_Expenses'],
                "Disposable_Income": row['Disposable_Income'],
                "Savings_Gap": row['Savings_Gap'],
                "Expense_Ratio": round(row['Expense_Ratio'], 4),
            },
            "shap_data": shap_list
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
