import sklearn
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)


# Load the model and scaler
with open('diabetes_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('diabetes_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request form
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree = float(request.form['diabetes_pedigree'])
    age = float(request.form['age'])

    # Process input data
    input_data = np.array(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    result = 'Diabetic' if prediction[0] == 1 else 'Non-diabetic'

    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
