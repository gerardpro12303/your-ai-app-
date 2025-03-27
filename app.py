from flask import Flask, request, jsonify, render_template
import pickle
import random
import pandas as pd
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load pre-trained model and transformer
try:
    model = pickle.load(open("model.pkl", "rb"))
    column_transformer = pickle.load(open("column_transformer.pkl", "rb"))
except Exception as e:
    print(f"Error loading model or column transformer: {e}")
    raise

# Define responses
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

        # Log the incoming data for debugging
        print("Received data:", data)

        # Create DataFrame for incoming data - treating categorical features properly
        new_patient_df = pd.DataFrame({
            "Family_History": [int(data["Family_History"])],
            "Glucose_Reading": [float(data["Glucose_Reading"])],
            "Frequent_Urination": [int(data["Frequent_Urination"])],
            "Fatigue": [int(data["Fatigue"])],
            "Blurred_Vision": [int(data["Blurred_Vision"])],
            "Age": [int(data["Age"])],
            "Diet_Quality": [data["Diet_Quality"]],  # Leave as string, column transformer will handle it
            "Gender": [data["Gender"]]  # Leave as string, column transformer will handle it
        })

        # Transform the data using the pre-fitted column transformer
        new_patient_encoded = column_transformer.transform(new_patient_df)
        feature_names = column_transformer.get_feature_names_out()

        # Log the transformed data
        print("Transformed data:", new_patient_encoded)

        # Create a DataFrame with the transformed data
        new_patient_encoded_df = pd.DataFrame(new_patient_encoded, columns=feature_names)

        # Make prediction using the pre-fitted model
        prediction = model.predict(new_patient_encoded_df)[0]
        prediction_proba = model.predict_proba(new_patient_encoded_df)
     
        print("Prediction:", prediction, "Confidence:", prediction_proba)

        # Retrieve the actual diagnosis (Must be provided in the input)
        actual_label = int(data["Actual_Label"])  # Add this field in the input data

        # Compute accuracy for this patient
        patient_accuracy = 1.0 if prediction == actual_label else 0.0

        # Compute precision (only applies if prediction == 1)
        if prediction == 1:
            true_positive = 1 if actual_label == 1 else 0
            false_positive = 1 if actual_label == 0 else 0
            patient_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        else:
            patient_precision = "N/A"  # Precision is not applicable for negative predictions

        # Return results
        result_text = random.choice(high_risk_messages) if prediction == 1 else random.choice(low_risk_messages)

        return jsonify({
            "prediction": result_text,
            "confidence": prediction_proba.tolist(),
            "accuracy_for_patient": round(patient_accuracy, 4),
            "precision_for_patient": patient_precision if isinstance(patient_precision, str) else round(patient_precision, 4)
        })

    except Exception as e:
        print(f"Error occurred during prediction: {e}")
        return jsonify({"error": str(e)}), 400

        # Log prediction and confidence levels
        print("Prediction:", prediction, "Confidence:", prediction_proba)

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
        print(f"Error occurred during prediction: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)


