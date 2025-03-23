document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("prediction-form");
    const resultDiv = document.getElementById("result");

    form.addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent page reload

        // Get form data
        let formData = new FormData(form);

        // Send data to Flask using Fetch API
        fetch("/", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            resultDiv.innerHTML = `<h3>Prediction: ${data.prediction}</h3>`;
        })
        .catch(error => console.error("Error:", error));
    });
});
