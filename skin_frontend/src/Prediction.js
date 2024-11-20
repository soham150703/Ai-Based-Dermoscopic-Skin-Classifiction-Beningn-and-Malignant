import React, { useState } from "react";

function Prediction() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null); // State for image preview
  const [prediction, setPrediction] = useState(""); // State to store prediction result
  const [confidence, setConfidence] = useState(""); // State to store confidence level
  const [error, setError] = useState(""); // State to store error message
  const [feedbackShown, setFeedbackShown] = useState(false); // State to show feedback question
  const [classificationCorrect, setClassificationCorrect] = useState(null); // State to track user feedback

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPrediction(""); // Clear previous prediction
    setConfidence(""); // Clear previous confidence
    setError(""); // Clear previous error
    setFeedbackShown(false); // Reset feedback question
    setClassificationCorrect(null); // Reset classification response

    if (file) {
      setImagePreviewUrl(URL.createObjectURL(file)); // Generate a preview URL for the image
    }
  };

  const handleSubmit = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append("image", selectedFile);

      try {
        const response = await fetch("http://localhost:5000/predict", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          console.log(
            "Prediction result:",
            result.prediction,
            "Confidence:",
            result.confidence
          );

          setPrediction(result.prediction);
          setConfidence(result.confidence);
          setFeedbackShown(true); // Show feedback question
        } else {
          const errorResponse = await response.json();
          setError(errorResponse.error || "Prediction failed.");
        }
      } catch (error) {
        setError("An error occurred during prediction");
        console.error("Error during prediction:", error);
      }
    } else {
      setError("Please upload an image");
    }
  };

  const handleFeedback = (response) => {
    setClassificationCorrect(response);
    if (!response) {
      window.location.href = "/form"; // Redirect to form if the classification is incorrect
    }
  };

  return (
    <div>
      <h2>Upload an Image for Prediction</h2>
      <input type="file" onChange={handleFileChange} />

      {/* Display image preview */}
      {imagePreviewUrl && (
        <img
          src={imagePreviewUrl}
          alt="Uploaded"
          style={{ maxWidth: "300px", marginTop: "20px" }}
        />
      )}

      <button onClick={handleSubmit}>Predict</button>

      {/* Display prediction result */}
      {prediction && <h3>Prediction: {prediction}</h3>}
      {confidence && <h4>Confidence: {confidence.toFixed(2)}%</h4>}

      {/* Display error message */}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {/* Feedback question */}
      {feedbackShown && classificationCorrect === null && (
        <div>
          <h4>Was this classification correct?</h4>
          <button onClick={() => handleFeedback(true)}>Yes</button>
          <button onClick={() => handleFeedback(false)}>No</button>
        </div>
      )}

      {/* Confirmation of feedback */}
      {classificationCorrect === true && <p>Thank you for your feedback!</p>}
    </div>
  );
}

export default Prediction;
