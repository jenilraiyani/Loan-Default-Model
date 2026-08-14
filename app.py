from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS so the React frontend (running on a different port) can talk to this API
CORS(app)

# Load the saved model and preprocessing tools
# Ensure these files are in the exact same directory as this app.py file
try:
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    print("Model and preprocessing tools loaded successfully!")
except Exception as e:
    print(f"Error loading files. Make sure loan_model.pkl, scaler.pkl, and label_encoder.pkl exist. Error: {e}")

@app.route('/')
def home():
    return "Loan Default Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the JSON data sent from the React frontend
        data = request.get_json()

        # 2. Convert the incoming JSON dictionary into a Pandas DataFrame
        # We wrap 'data' in a list because pandas expects a list of records
        df_input = pd.DataFrame([data])

        # Ensure all columns are in lowercase and stripped of spaces (just like your training data)
        df_input.columns = df_input.columns.str.strip().str.lower().str.replace(' ', '_')

        # 3. Define the column groups
        numeric_cols = ['age', 'income', 'loanamount', 'creditscore', 'monthsemployed', 
                        'numcreditlines', 'interestrate', 'loanterm', 'dtiratio']
        
        categorical_cols = ['education', 'employmenttype', 'maritalstatus', 
                            'hasmortgage', 'hasdependents', 'loanpurpose', 'hascosigner']

        # 4. Ensure the columns are in the exact order the model was trained on
        # This matches the column order from your Task 2 DataFrame
        expected_columns = numeric_cols + categorical_cols
        
        # Add any missing columns as 0 to prevent crashes (defensive programming)
        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        # Reorder the dataframe to match training
        df_input = df_input[expected_columns]

        # 5. Preprocess Categorical Data
        # We try to use the loaded LabelEncoder. If the input is string, transform it.
        for col in categorical_cols:
            if df_input[col].dtype == 'O' or isinstance(df_input[col].iloc[0], str):
                try:
                    df_input[col] = le.transform(df_input[col])
                except ValueError:
                    # Fallback: if the user sends a category the encoder hasn't seen, default to 0
                    df_input[col] = 0 

        # 6. Preprocess Numeric Data
        # Apply the loaded MinMaxScaler to the numeric columns
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

        # 7. Make the Prediction
        # predict() returns an array like [0] or [1]
        prediction_array = model.predict(df_input.values)
        prediction_result = int(prediction_array[0])

        # Determine the human-readable risk level
        risk_status = "High Risk (Default)" if prediction_result == 1 else "Low Risk (No Default)"

        # 8. Send the result back to React as JSON
        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'risk_status': risk_status
        })

    except Exception as e:
        # If anything goes wrong, return the error message securely
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Start the server
if __name__ == '__main__':
    # Runs on http://127.0.0.1:5000 by default
    app.run(debug=True, port=5000)