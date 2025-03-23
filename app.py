from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import random

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Lists of different responses for variation
high_risk_messages = [
    "⚠️ High Risk of Diabetes Detected! Please consult a doctor immediately.",
    "🩺 Your results indicate a high risk of diabetes. Medical consultation is advised.",
    "🚨 Warning: You may be at risk for diabetes. Early detection is key, seek medical help.",
    "❗ Custom Message 1",
    "❗ Custom Message 2"
]

low_risk_messages = [
    "✅ No immediate risk detected. Keep up your healthy habits!",
    "🌟 Your results show no signs of diabetes. Stay active and eat well!",
    "👍 No diabetes risk detected for now. Maintain a balanced lifestyle.",
    "💚 Custom Message 1",
    "💚 Custom Message 2"
]

high_risk_advice = [
    "💡 Tips: Cut down on processed sugar, eat fiber-rich food, and stay hydrated.",
    "🍏 Try incorporating more vegetables and lean protein into your diet.",
    "🏃‍♂️ Regular physical activity and routine check-ups can help manage your risk.",
    "🛑 Custom Advice 1",
    "🛑 Custom Advice 2"
]

low_risk_advice = [
    "💡 Keep up the good work! Exercise and proper diet are key to long-term health.",
    "🥗 Consider adding more whole foods to your meals for better nutrition.",
    "🩺 Even without risk now, routine check-ups help prevent future problems.",
    "🌿 Custom Advice 1",
    "🌿 Custom Advice 2"
]

# Define the feature columns
categorical_features = ["Gender", "Diet_Quality"]
numerical_features = ["Family_History", "Glucose_Reading", "Frequent_Urination", "Fatigue", "Blurred_Vision", "Age"]

# 🔹 Define the column transformer (this is where the error occurred)
column_transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

# Load training data (this should match how you trained your model)
try:
    X_train = pickle.load(open("X_train.pkl", "rb"))  # Load training data if available
    column_transformer.fit(X_train)  # Fit transformer on training data
except FileNotFoundError:
    print("Warning: X_train.pkl not found. Make sure the transformer is properly trained.")

# Define and fit the scaler (only if needed)
scaler = StandardScaler()

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")  # Ensure this exists

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:  
            data = request.json  
        else:  
            data = request.form.to_dict()  

        # 🔹 Define input as a DataFrame
        new_patient = pd.DataFrame({
            "Family_History": [int(data["Family_History"])],
            "Glucose_Reading": [float(data["Glucose_Reading"])],
            "Frequent_Urination": [int(data["Frequent_Urination"])],
            "Fatigue": [int(data["Fatigue"])],
            "Blurred_Vision": [int(data["Blurred_Vision"])],
            "Age": [int(data["Age"])],
            "Diet_Quality": [data["Diet_Quality"]],
            "Gender": [data["Gender"]]
        })

      # ✅ Step 3: Apply the **loaded** column transformer
        new_patient_encoded = column_transformer.transform(new_patient_df)
        feature_names = column_transformer.get_feature_names_out()
        new_patient_encoded_df = pd.DataFrame(new_patient_encoded, columns=feature_names)

        # ✅ Scale the transformed data using the **loaded** scaler
        new_patient_scaled = scaler.transform(new_patient_encoded_df)  # Use transform() instead of fit_transform()
        new_patient_scaled_df = pd.DataFrame(new_patient_scaled, columns=feature_names)

        # 🔹 Make a prediction
        prediction = model.predict(new_patient_scaled_df)[0]
        prediction_proba = model.predict_proba(new_patient_scaled_df)

        # 🔹 Interpret the result
        if prediction == 1:
            result_text = random.choice(high_risk_messages)
            advice = random.choice(high_risk_advice)
        else:
            result_text = random.choice(low_risk_messages)
            advice = random.choice(low_risk_advice)

        return jsonify({
            "prediction": result_text,
            "advice": advice,
            "confidence": prediction_proba.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)


