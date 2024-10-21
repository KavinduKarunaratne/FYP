from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('stroke_prediction_model.pkl', 'rb'))

app = Flask(__name__)

# Helper functions for binning
def bin_blood_pressure(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return 0  # Normal
    elif 120 <= systolic < 130 and diastolic < 80:
        return 1  # Elevated
    elif (130 <= systolic < 140 or 80 <= diastolic < 90):
        return 2  # Hypertension Stage 1
    elif (140 <= systolic < 180 or 90 <= diastolic < 120):
        return 3  # Hypertension Stage 2
    elif systolic >= 180 or diastolic >= 120:
        return 4  # Hypertensive Crisis
    else:
        return 5  # Unknown

def bin_hdl(hdl):
    if hdl < 40:
        return 0  # Low (Poor)
    elif 40 <= hdl < 60:
        return 1  # Normal
    else:
        return 2  # High (Good)

def bin_ldl(ldl):
    if ldl < 100:
        return 0  # Optimal
    elif 100 <= ldl < 130:
        return 1  # Near Optimal/Above Optimal
    elif 130 <= ldl < 160:
        return 2  # Borderline High
    elif 160 <= ldl < 190:
        return 3  # High
    else:
        return 4  # Very High

def bin_glucose_level(glucose):
    if glucose < 140:
        return 0  # Normal
    elif 140 <= glucose < 200:
        return 1  # Prediabetes
    else:
        return 2  # Diabetes

def bin_bmi(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif 18.5 <= bmi < 25:
        return 1  # Normal weight
    elif 25 <= bmi < 30:
        return 2  # Overweight
    elif 30 <= bmi < 35:
        return 3  # Moderate Obesity
    elif 35 <= bmi < 40:
        return 4  # Severe Obesity
    else:
        return 5  # Morbid Obesity

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Collect form data into a dictionary
        form_data = {
            'age': request.form['age'],
            'gender': request.form['gender'],
            'hypertension': request.form['hypertension'],
            'heart_disease': request.form['heart_disease'],
            'marital_status': request.form['marital_status'],
            'work_type': request.form['work_type'],
            'residence_type': request.form['residence_type'],
            'glucose': float(request.form['glucose']),
            'bmi': float(request.form['bmi']),
            'smoking_status': request.form['smoking_status'],
            'alcohol_intake': request.form['alcohol_intake'],
            'physical_activity': request.form['physical_activity'],
            'stroke_history': request.form['stroke_history'],
            'family_history_stroke': request.form['family_history_stroke'],
            'dietary_habits': request.form['dietary_habits'],
            'stress_level': request.form['stress_level'],
            'blood_pressure': request.form['blood_pressure'],
            'hdl': request.form['hdl'],
            'ldl': request.form['ldl'],
            'blurred_vision': 'blurred_vision' in request.form,
            'confusion': 'confusion' in request.form,
            'difficulty_speaking': 'difficulty_speaking' in request.form,
            'dizziness': 'dizziness' in request.form,
            'headache': 'headache' in request.form,
            'loss_of_balance': 'loss_of_balance' in request.form,
            'numbness': 'numbness' in request.form,
            'seizures': 'seizures' in request.form,
            'severe_fatigue': 'severe_fatigue' in request.form,
            'weakness': 'weakness' in request.form
        }

        # Prepare data for prediction (ensure correct format)
        features = [
            int(form_data['age']),
            int(form_data['gender']),
            int(form_data['hypertension']),
            int(form_data['heart_disease']),
            int(form_data['marital_status']),
            int(form_data['work_type']),
            int(form_data['residence_type']),
            float(form_data['glucose']),
            float(form_data['bmi']),
            int(form_data['smoking_status']),
            int(form_data['alcohol_intake']),
            int(form_data['physical_activity']),
            int(form_data['stroke_history']),
            int(form_data['family_history_stroke']),
            int(form_data['dietary_habits']),
            int(form_data['stress_level']),
            int(form_data['blood_pressure']),
            float(form_data['hdl']),
            float(form_data['ldl']),
            int(form_data['blurred_vision']),
            int(form_data['confusion']),
            int(form_data['difficulty_speaking']),
            int(form_data['dizziness']),
            int(form_data['headache']),
            int(form_data['loss_of_balance']),
            int(form_data['numbness']),
            int(form_data['seizures']),
            int(form_data['severe_fatigue']),
            int(form_data['weakness'])
        ]

        # Model prediction
        prediction = model.predict([features])[0]

        # Convert numerical prediction to label
        if prediction == 1:
            prediction = "Stroke"
        else:
            prediction = "No Stroke"

        # Render the template with prediction and form data
        return render_template('index.html', prediction=prediction, **form_data)

    return render_template('index.html', prediction="Not Determined")

if __name__ == '__main__':
    app.run(debug=True)
