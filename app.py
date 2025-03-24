from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import random
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

# Load the trained model and transformers (this assumes the model, column_transformer, and scaler are already saved)
model = pickle.load(open("model.pkl", "rb"))
column_transformer = pickle.load(open("column_transformer.pkl", "rb"))

# Define the feature columns
categorical_features = ["Gender", "Diet_Quality"]
numerical_features = ["Family_History", "Glucose_Reading", "Frequent_Urination", "Fatigue", "Blurred_Vision", "Age"]

# Sample training data for column transformer fitting
X_train = pd.DataFrame({
    "Family_History": [0, 1, 0, 1, 0, 1, 0],
    "Glucose_Reading": [100, 150, 99, 50, 60, 70, 80],
    "Frequent_Urination": [0, 1, 0, 1, 0, 1, 0],
    "Fatigue": [0, 1, 0, 1, 0, 1, 0],
    "Blurred_Vision": [0, 1, 0, 1, 0, 1, 0],
    "Age": [25, 30, 20, 40, 50, 60, 70],
    "Diet_Quality": ['Good', 'Average', 'Poor', 'Average', 'Poor', 'Average', 'Poor'],
    "Gender": ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
})

# Fit transformers on the training data at the start of the app
column_transformer.fit(X_train)

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

        # Create a DataFrame for the incoming data
        new_patient_df = pd.DataFrame({
             "Family_History": [int(data["Family_History"])],
             "Glucose_Reading": [float(data["Glucose_Reading"])],
             "Frequent_Urination": [int(data["Frequent_Urination"])],
             "Fatigue": [int(data["Fatigue"])],
             "Blurred_Vision": [int(data["Blurred_Vision"])],
             "Age": [int(data["Age"])],
             "Diet_Quality": [data["Diet_Quality"]],  # Ensure the value is 'Good', 'Poor', or 'Average'
             "Gender": [data["Gender"]]  # Ensure valid values 'Male' or 'Female'
        })

        # Transform the data using the fitted column transformer (this applies one-hot encoding to categorical data)
        new_patient_encoded = column_transformer.transform(new_patient_df)
        feature_names = column_transformer.get_feature_names_out()

        # Ensure that the DataFrame is properly constructed with the correct columns
        new_patient_encoded_df = pd.DataFrame(new_patient_encoded, columns=feature_names)

        # Make prediction using the pre-fitted model
        prediction = model.predict(new_patient_encoded_df)[0]
        prediction_proba = model.predict_proba(new_patient_encoded_df)

        # Return results based on the prediction
        if prediction == 1:
            result_text = random.choice(high_risk_messages)
        else:
            result_text = random.choice(low_risk_messages)

        return jsonify({
            "prediction": result_text,
            "confidence": prediction_proba.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)


