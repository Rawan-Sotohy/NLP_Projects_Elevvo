// Get HTML elements
const predictBtn = document.getElementById("predictBtn");
const inputText = document.getElementById("inputText");
const resultCard = document.getElementById("resultCard");
const categoryText = document.getElementById("categoryText");

// Event listener for Predict button
predictBtn.addEventListener("click", async () => {
    const text = inputText.value;
    if (!text.trim()) {
        alert("Please enter some text!");
        return;
    }

    // Send POST request to /predict
    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    const data = await response.json();
    if (data.error) {
        alert(data.error);
        return;
    }

    // Show the predicted category
    categoryText.innerText = data.category;
    resultCard.classList.remove("hidden");

    // Apply color styling based on category
    setCategoryColor(data.category);
});

// Function to set colors for each category
function setCategoryColor(category) {
    const resultCard = document.getElementById("resultCard");
    const categoryText = document.getElementById("categoryText");
    let color = "#333"; // default

    switch (category) {
        case "World":
            color = "#007bff"; 
            break;
        case "Sports":
            color = "#28a745"; 
            break;
        case "Business":
            color = "#f1a027ff"; 
            break;
        case "Sci/Tech":
            color = "#6f42c1"; 
            break;
    }

    categoryText.style.color = color;
    resultCard.style.borderTop = `10px solid ${color}`;
}
