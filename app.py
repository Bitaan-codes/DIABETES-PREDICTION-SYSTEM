import gradio as gr
import numpy as np
import joblib

model = joblib.load("diabetes_prediction_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

def predict_diabetes(
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
):
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "Diabetic"
    else:
        return "Non-Diabetic"

diabetes_prediction_app = gr.Interface(
    fn = predict_diabetes,
    inputs = [
        gr.Number(label = "Pregnancies"),
        gr.Number(label = "Glucose"),
        gr.Number(label = "Blood Pressure"),
        gr.Number(label = "Skin Thickness"),
        gr.Number(label = "Insulin"),
        gr.Number(label = "BMI"),
        gr.Number(label = "Diabetes Pedigree Function"),
        gr.Number(label = "Age"),
    ],
        outputs = gr.Textbox(label = "Prediction"),
        title = "Diabetes Prediction System",
        description = "Predict whether a patient is Diabetic or Non-Diabetic",
)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))

    diabetes_prediction_app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
