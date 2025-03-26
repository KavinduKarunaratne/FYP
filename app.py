from flask import Flask, render_template, request
from xgboost import XGBClassifier
import pandas as pd

# Load the trained model
model = XGBClassifier()
model.load_model('stroke_prediction_model.json')

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Collect form data and convert to the correct type
        try:
            form_data = {
                'age': int(request.form['age']),
                'hypertension': int(request.form['hypertension']),
                'heart_disease': int(request.form['heart_disease']),
                'glucose': float(request.form['glucose']),
                'bmi': float(request.form['bmi']),
                'smoking_status': int(request.form['smoking_status']),
                'alcohol_intake': int(request.form['alcohol_intake']),
                'physical_activity': int(request.form['physical_activity']),
                'stroke_history': int(request.form['stroke_history']),
                'stress_level': int(request.form['stress_level']),
                'systolic': int(request.form['systolic']),
                'diastolic': int(request.form['diastolic']),
                'hdl': int(request.form['hdl']),
                'ldl': int(request.form['ldl']),
                'gender': int(request.form['gender']),
            }

            # Map form field names to model's expected feature names
            feature_mapping = {
                'age': 'Age',
                'hypertension': 'Hypertension',
                'heart_disease': 'Heart Disease',
                'glucose': 'Average Glucose Level',
                'bmi': 'Body Mass Index (BMI)',
                'smoking_status': 'Smoking Status',
                'alcohol_intake': 'Alcohol Intake',
                'physical_activity': 'Physical Activity',
                'stroke_history': 'Stroke History',
                'stress_level': 'Stress Levels',
                'systolic': 'Systolic_BP',
                'diastolic': 'Diastolic_BP',
                'hdl': 'HDL',
                'ldl': 'LDL',
                'gender': 'Gender_Male'
            }

            # Apply mapping to match model's feature names
            model_input = {feature_mapping[key]: value for key, value in form_data.items()}


            # Convert to DataFrame
            input_df = pd.DataFrame([model_input])


            # Ensure model input has the same order of columns the model was trained on
            expected_columns = [
                'Age', 'Hypertension', 'Heart Disease', 'Average Glucose Level',
                'Body Mass Index (BMI)', 'Smoking Status', 'Alcohol Intake', 'Physical Activity',
                'Stroke History', 'Stress Levels', 'Systolic_BP', 'Diastolic_BP',
                'HDL', 'LDL', 'Gender_Male'
            ]
            input_df = input_df[expected_columns]

            # Model prediction
            prediction = model.predict(input_df)[0]


            # Convert numerical prediction to label if needed
            if prediction == 1:
                result = "Stroke"
            else:
                result = "No Stroke"


            return render_template('index.html', prediction=result, **form_data)
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

    return render_template('index.html', prediction="Not Determined")

if __name__ == '__main__':
    app.run(debug=True)
