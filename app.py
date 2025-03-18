from flask import Flask, render_template, request
import joblib

# Load the trained model
model = joblib.load(open('stroke_prediction_model.pkl', 'rb'))

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Collect form data into a dictionary
        form_data = {
            'age': request.form['age'],
            'gender': request.form['gender'],
            'hypertension': request.form['hypertension'],
            'heart_disease': request.form['heart_disease'],
            'glucose': float(request.form['glucose']),
            'bmi': float(request.form['bmi']),
            'smoking_status': request.form['smoking_status'],
            'alcohol_intake': request.form['alcohol_intake'],
            'physical_activity': request.form['physical_activity'],
            'stroke_history': request.form['stroke_history'],
            'stress_level': request.form['stress_level'],
            'systolic': request.form['systolic'],
            'diastolic': request.form['diastolic'],
            'hdl': request.form['hdl'],
            'ldl': request.form['ldl'],
        }

        # Prepare data for prediction (ensure correct format)
        features = [
            int(form_data['age']),
            int(form_data['gender']),
            int(form_data['hypertension']),
            int(form_data['heart_disease']),
            float(form_data['glucose']),
            float(form_data['bmi']),
            int(form_data['smoking_status']),
            int(form_data['alcohol_intake']),
            int(form_data['physical_activity']),
            int(form_data['stroke_history']),
            int(form_data['stress_level']),
            int(form_data['systolic']),
            int(form_data['diastolic']),
            int(form_data['hdl']),
            int(form_data['ldl']),
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
