from flask import Flask, request, render_template, jsonify
import random  # Import random for varied responses

app = Flask(__name__)

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
        # Simulating prediction logic (replace with real prediction)
        prediction_result = random.choice([1, 0])  # Randomly simulate risk levels

        if prediction_result == 1:
            result_text = random.choice(high_risk_messages)
            advice = random.choice(high_risk_advice)
        else:
            result_text = random.choice(low_risk_messages)
            advice = random.choice(low_risk_advice)

        return jsonify({"prediction": result_text, "advice": advice})  # Return JSON response

    return render_template("index.html")  # Show the home page

if __name__ == "__main__":
    app.run(debug=True)
