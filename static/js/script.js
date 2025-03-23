<script>
document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("prediction-form").addEventListener("submit", async function (event) {
        event.preventDefault(); // Stop page reload

        // Collect input values
        let formData = {
            "Family_History": parseInt(document.querySelector("input[name='family_history']").value),
            "Glucose_Reading": parseFloat(document.querySelector("input[name='glucose_reading']").value),
            "Frequent_Urination": parseInt(document.querySelector("input[name='frequent_urination']").value),
            "Fatigue": parseInt(document.querySelector("input[name='fatigue']").value),
            "Blurred_Vision": parseInt(document.querySelector("input[name='blurred_vision']").value),
            "Age": parseInt(document.querySelector("input[name='age']").value),
            "Diet_Quality": document.querySelector("select[name='diet_quality']").value,
            "Gender": document.querySelector("select[name='gender']").value
        };

        try {
            let response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData)
            });

            let result = await response.json();
            document.getElementById("result").innerHTML = `<h3>Prediction: ${result.prediction}</h3>`;
        } catch (error) {
            console.error("Error:", error);
            document.getElementById("result").innerHTML = `<h3 style="color: red;">Error predicting</h3>`;
        }
    });
});
</script>
