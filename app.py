from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import random
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

# Load the trained model and column transformer (this assumes model.pkl and column_transformer.pkl exist in the same directory)
model = pickle.load(open("model.pkl", "rb"))
column_transformer = pickle.load(open("column_transformer.pkl", "rb"))

# Lists of responses for variation
high_risk_messages = [
    "⚠️ High Risk of Diabetes Detected! Please consult a doctor immediately.",
    "🩺 Your results indicate a high risk of diabetes. Medical consultation is advised.",
    "🚨 Warning: You may be at risk for diabetes. Early detection is key, seek medical help."
]

low_risk_messages = [
    "✅ No immediate risk detected. Keep up your healthy habits!",
    "🌟 Your results show no signs of diabetes. Stay active and eat well!"
]

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()

        # Ensure categorical values are correctly processed
        diet_quality = data.get("Diet_Quality", "Good")  # Default to 'Good' if not provided
        gender = data.get("Gender", "Male")  # Default to 'Male' if not provided

        # Create a DataFrame for the incoming data
        new_patient_df = pd.DataFrame({
             "Family_History": [int(data["Family_History"])],
             "Glucose_Reading": [float(data["Glucose_Reading"])],
             "Frequent_Urination": [int(data["Frequent_Urination"])],
             "Fatigue": [int(data["Fatigue"])],
             "Blurred_Vision": [int(data["Blurred_Vision"])],
             "Age": [int(data["Age"])],
             "Diet_Quality": [diet_quality],  # Ensure the value is 'Good', 'Poor', or 'Average'
             "Gender": [gender]  # Ensure valid values 'Male' or 'Female'
        })

        # Transform the data (do not call fit again, just transform)
        new_patient_encoded = column_transformer.transform(new_patient_df)
        feature_names = column_transformer.get_feature_names_out()

        # Ensure that the DataFrame is properly constructed with the correct columns
        new_patient_encoded_df = pd.DataFrame(new_patient_encoded, columns=feature_names)

        # Make prediction using the pre-fitted model
        prediction = model.predict(new_patient_encoded_df)[0]
        prediction_proba = model.predict_proba(new_patient_encoded_df)[0]  # Get probabilities for both classes

        # Return results based on the prediction
        if prediction == 1:
            result_text = random.choice(high_risk_messages)
        else:
            result_text = random.choice(low_risk_messages)

        return jsonify({
            "prediction": result_text,
            "confidence": prediction_proba.tolist()  # Return the probabilities for both classes
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

