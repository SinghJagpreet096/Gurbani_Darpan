function sendToBackend() {
    let userText = document.getElementById("userInput").value;

    // Show the spinner while waiting
    document.getElementById("waitingSpinner").style.display = "inline-block";

    fetch("http://127.0.0.1:5001/response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText })
    })
    .then(response => response.json())
    .then(data => {
        // Hide the spinner once the response is received
        document.getElementById("waitingSpinner").style.display = "none";
        
        document.getElementById("responseText").innerText = "Response: " + data.response;
    })
    .catch(error => console.error("Error:", error));
}
