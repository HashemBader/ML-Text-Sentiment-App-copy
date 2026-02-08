const reviewText = document.getElementById("reviewText");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");

const setLoading = (isLoading) => {
  predictBtn.disabled = isLoading;
  predictBtn.textContent = isLoading ? "Predicting..." : "Predict";
};

const updateResult = ({ label, sentiment, confidence }) => {
  resultLabel.textContent = label || "Awaiting input...";
  resultLabel.classList.remove("good", "bad");
  if (sentiment === "good") {
    resultLabel.classList.add("good");
  } else if (sentiment === "bad") {
    resultLabel.classList.add("bad");
  }
  if (typeof confidence === "number") {
    resultConfidence.textContent = `Confidence: ${confidence.toFixed(1)}%`;
  } else {
    resultConfidence.textContent = "";
  }
  resultLabel.animate(
    [
      { transform: "translateY(4px)", opacity: 0.6 },
      { transform: "translateY(0)", opacity: 1 },
    ],
    { duration: 220, easing: "ease-out" }
  );
};

predictBtn.addEventListener("click", async () => {
  const text = reviewText.value.trim();
  if (!text) {
    updateResult({ label: "Please enter a review.", sentiment: "neutral" });
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    if (!response.ok) {
      updateResult({ label: data.error || "Prediction failed.", sentiment: "neutral" });
    } else {
      updateResult(data);
    }
  } catch (error) {
    updateResult({ label: "Network error. Please retry.", sentiment: "neutral" });
  } finally {
    setLoading(false);
  }
});

clearBtn.addEventListener("click", () => {
  reviewText.value = "";
  updateResult({ label: "Awaiting input...", sentiment: "neutral" });
});
