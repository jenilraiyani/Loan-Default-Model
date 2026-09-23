from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_joblib(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

logistic_model = load_joblib('logistic_model.pkl') or load_joblib('loan_model.pkl')
rf_model = load_joblib('rf_model.pkl') or load_joblib('loan_model.pkl')
scaler = load_joblib('scaler.pkl')
le = load_joblib('label_encoder.pkl')

print('Logistic:', type(logistic_model).__name__ if logistic_model is not None else 'MISSING')
print('Random Forest:', type(rf_model).__name__ if rf_model is not None else 'MISSING')
print('Scaler:', 'loaded' if scaler is not None else 'MISSING')

@app.route('/')
def home():
    return "Loan Default Prediction API is running!"

def make_prediction(model, df_input):
    values = df_input.to_numpy()
    prediction_result = int(model.predict(values)[0])
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = round(float(model.predict_proba(values)[0][1]), 4)
    return {
        'prediction': prediction_result,
        'probability': probability,
        'risk_status': "High Risk (Default)" if prediction_result == 1 else "Low Risk (No Default)"
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if scaler is None:
            return jsonify({'success': False, 'error': 'scaler.pkl is missing in the backend folder'}), 400

        data = request.get_json() or {}
        model_choice = str(data.pop('model', 'both')).strip().lower().replace('-', '_')
        if model_choice in ('logisticregression', 'lr'):
            model_choice = 'logistic'
        if model_choice in ('rf', 'randomforest'):
            model_choice = 'random_forest'
        if model_choice not in ('logistic', 'random_forest', 'both'):
            model_choice = 'both'

        df_input = pd.DataFrame([data])
        df_input.columns = df_input.columns.str.strip().str.lower().str.replace(' ', '_')

        numeric_cols = ['age', 'income', 'loanamount', 'creditscore', 'monthsemployed',
                        'numcreditlines', 'interestrate', 'loanterm', 'dtiratio']
        categorical_cols = ['education', 'employmenttype', 'maritalstatus',
                            'hasmortgage', 'hasdependents', 'loanpurpose', 'hascosigner']
        expected_columns = numeric_cols + categorical_cols

        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[expected_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        if le is not None:
            for col in categorical_cols:
                if df_input[col].dtype == 'O' or isinstance(df_input[col].iloc[0], str):
                    try:
                        df_input[col] = le.transform(df_input[col])
                    except ValueError:
                        df_input[col] = 0

        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

        response = {
            'success': True,
            'selected_model': model_choice
        }

        if model_choice in ('logistic', 'both'):
            if logistic_model is None:
                return jsonify({'success': False, 'error': 'logistic_model.pkl not found'}), 400
            logistic_result = make_prediction(logistic_model, df_input)
            response['logistic_regression'] = {
                'name': 'Logistic Regression',
                **logistic_result
            }

        if model_choice in ('random_forest', 'both'):
            if rf_model is None:
                return jsonify({'success': False, 'error': 'rf_model.pkl not found'}), 400
            rf_result = make_prediction(rf_model, df_input)
            response['random_forest'] = {
                'name': 'Random Forest',
                **rf_result
            }

        primary = response.get('random_forest') or response.get('logistic_regression')
        response['prediction'] = primary['prediction']
        response['risk_status'] = primary['risk_status']
        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
