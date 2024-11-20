import React, { useState } from "react";
import axios from "axios";

function Prediction2() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [imagePreview, setImagePreview] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setImagePreview(URL.createObjectURL(file));
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      alert("Please upload an image");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://localhost:8000/predict2", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log("Backend Response:", response.data);
      if (response.data.prediction !== undefined) {
        setPrediction(response.data.prediction);
      } else {
        alert("Prediction failed. Backend did not return a valid response.");
      }
    } catch (error) {
      console.error("Error during prediction:", error.response || error);
      alert("An error occurred while processing the image.");
    }
  };

  return (
    <div style={{ textAlign: "center", margin: "20px" }}>
      <h1>ISIC Model Classification</h1>
      <p>Upload an image for classification using the secondary model.</p>

      <input type="file" accept="image/*" onChange={handleFileChange} />
      <br />
      {imagePreview && <img src={imagePreview} alt="Uploaded" style={{ marginTop: "20px", width: "300px" }} />}
      <br />

      <label htmlFor="threshold">Classification Threshold: {threshold}</label>
      <input
        id="threshold"
        type="range"
        min="0"
        max="1"
        step="0.01"
        value={threshold}
        onChange={(e) => setThreshold(parseFloat(e.target.value))}
      />
      <br />

      <button onClick={handleSubmit} style={{ marginTop: "20px" }}>
        Predict
      </button>

      {prediction !== null && (
        <div style={{ marginTop: "20px" }}>
          <h2>Prediction Score: {prediction.toFixed(4)}</h2>
          {prediction >= threshold ? (
            <h3 style={{ color: "red" }}>Prediction: Malignant (High Risk)</h3>
          ) : (
            <h3 style={{ color: "green" }}>Prediction: Benign (Low Risk)</h3>
          )}
        </div>
      )}
    </div>
  );
}

export default Prediction2;
