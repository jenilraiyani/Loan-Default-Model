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

MODELS = {
    'logistic': {
        'name': 'Logistic Regression',
        'response_key': 'logistic_regression',
        'model': load_joblib('logistic_model.pkl') or load_joblib('loan_model.pkl'),
    },
    'decision_tree': {
        'name': 'Decision Tree',
        'response_key': 'decision_tree',
        'model': load_joblib('decision_tree_model.pkl'),
    },
    'naive_bayes': {
        'name': 'Naive Bayes',
        'response_key': 'naive_bayes',
        'model': load_joblib('naive_bayes_model.pkl'),
    },
    'random_forest': {
        'name': 'Random Forest',
        'response_key': 'random_forest',
        'model': load_joblib('rf_model.pkl') or load_joblib('loan_model.pkl'),
    },
    'adaboost': {
        'name': 'AdaBoost',
        'response_key': 'adaboost',
        'model': load_joblib('adaboost_model.pkl'),
    },
}

scaler = load_joblib('scaler.pkl')
le = load_joblib('label_encoder.pkl')

for key, meta in MODELS.items():
    loaded = meta['model'] is not None
    print(f"{meta['name']}: {type(meta['model']).__name__ if loaded else 'MISSING'}")
print('Scaler:', 'loaded' if scaler is not None else 'MISSING')

ALIASES = {
    'logisticregression': 'logistic',
    'lr': 'logistic',
    'rf': 'random_forest',
    'randomforest': 'random_forest',
    'dt': 'decision_tree',
    'decisiontree': 'decision_tree',
    'nb': 'naive_bayes',
    'naivebayes': 'naive_bayes',
    'ada': 'adaboost',
    'both': 'all',
}

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
        model_choice = str(data.pop('model', 'all')).strip().lower().replace('-', '_').replace(' ', '_')
        model_choice = ALIASES.get(model_choice, model_choice)

        valid = set(MODELS.keys()) | {'all'}
        if model_choice not in valid:
            model_choice = 'all'

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

        selected_keys = list(MODELS.keys()) if model_choice == 'all' else [model_choice]
        response = {
            'success': True,
            'selected_model': model_choice
        }

        primary = None
        for key in selected_keys:
            meta = MODELS[key]
            if meta['model'] is None:
                return jsonify({'success': False, 'error': f"{meta['name']} model file not found"}), 400
            result = make_prediction(meta['model'], df_input)
            payload = {'name': meta['name'], **result}
            response[meta['response_key']] = payload
            primary = payload

        # Prefer Random Forest as primary when comparing all
        if model_choice == 'all' and 'random_forest' in response:
            primary = response['random_forest']

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
