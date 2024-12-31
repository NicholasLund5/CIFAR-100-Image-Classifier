document.getElementById("image-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const fileInput = document.getElementById("image-input");
    const resultDiv = document.getElementById("prediction-result");
    const imagePreview = document.getElementById("image-preview");

    if (!fileInput.files.length) {
        resultDiv.textContent = "Please select an image!";
        imagePreview.style.display = "none"; 
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();
    reader.onload = function (e) {
        imagePreview.src = e.target.result; 
        imagePreview.style.display = "block"; 
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("http://localhost:5000/classify/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to classify the image. Please try again.");
        }

        const data = await response.json();
        resultDiv.textContent = `Prediction: ${data.prediction}`;
    } catch (error) {
        console.error(error);
        resultDiv.textContent = "Error: Unable to classify the image.";
    }
});