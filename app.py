from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import random  # Import random for varied responses

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Lists of different responses for variation
high_risk_messages = [
    "⚠️ High Risk of Diabetes Detected! Please consult a doctor immediately.",
    "🩺 Your results indicate a high risk of diabetes. Medical consultation is advised.",
    "🚨 Warning: You may be at risk for diabetes. Early detection is key, seek medical help.",
    "❗ Custom Message 1",  # Add your own message here
    "❗ Custom Message 2"   # Add more if you want
]

low_risk_messages = [
    "✅ No immediate risk detected. Keep up your healthy habits!",
    "🌟 Your results show no signs of diabetes. Stay active and eat well!",
    "👍 No diabetes risk detected for now. Maintain a balanced lifestyle.",
    "💚 Custom Message 1",  # Add your own message here
    "💚 Custom Message 2"   # Add more if you want
]

high_risk_advice = [
    "💡 Tips: Cut down on processed sugar, eat fiber-rich food, and stay hydrated.",
    "🍏 Try incorporating more vegetables and lean protein into your diet.",
    "🏃‍♂️ Regular physical activity and routine check-ups can help manage your risk.",
    "🛑 Custom Advice 1",  # Add your own advice here
    "🛑 Custom Advice 2"   # Add more if you want
]

low_risk_advice = [
    "💡 Keep up the good work! Exercise and proper diet are key to long-term health.",
    "🥗 Consider adding more whole foods to your meals for better nutrition.",
    "🩺 Even without risk now, routine check-ups help prevent future problems.",
    "🌿 Custom Advice 1",  # Add your own advice here
    "🌿 Custom Advice 2"   # Add more if you want
]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        @app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:  # If request comes from JSON
            data = request.json
            features = np.array([
                data["Family_History"],
                data["Glucose_Reading"],
                data["Frequent_Urination"],
                data["Fatigue"],
                data["Blurred_Vision"],
                data["Age"],
            ]).reshape(1, -1)
        else:  # If request comes from form
            input_data = [float(x) for x in request.form.values()]
            features = np.array([input_data])

        # Make prediction using the trained model
        prediction = model.predict(features)[0]
        risk_level = int(prediction)  # Convert to integer (0 or 1)

        # Select message and advice based on prediction
        if risk_level == 1:
            result_text = random.choice(high_risk_messages)
            advice = random.choice(high_risk_advice)
        else:
            result_text = random.choice(low_risk_messages)
            advice = random.choice(low_risk_advice)

        return jsonify({"prediction": result_text, "advice": advice})  # Return JSON response

    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Handle errors (e.g., missing/invalid inputs)

    return render_template("index.html")  # Show the webpage


if __name__ == "__main__":
    app.run(debug=True)
