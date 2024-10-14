import React, { useState } from 'react';

function Prediction() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);  // State for image preview
  const [prediction, setPrediction] = useState('');  // State to store prediction result
  const [confidence, setConfidence] = useState('');  // State to store confidence level
  const [error, setError] = useState('');            // State to store error message

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPrediction('');  // Clear previous prediction
    setConfidence('');   // Clear previous confidence
    setError('');        // Clear previous error

    if (file) {
      // Generate a preview URL for the image
      setImagePreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('image', selectedFile);

      try {
        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          console.log('Prediction result:', result.prediction, 'Confidence:', result.confidence);  // Debugging

          // Update state with prediction and confidence result
          setPrediction(result.prediction);
          setConfidence(result.confidence);
        } else {
          const errorResponse = await response.json();
          setError(errorResponse.error || 'Prediction failed.');  // Handle server error message
        }
      } catch (error) {
        setError('An error occurred during prediction');  // Handle network errors
        console.error('Error during prediction:', error);
      }
    } else {
      setError('Please upload an image');
    }
  };

  return (
    <div>
      <h2>Upload an Image for Prediction</h2>
      <input type="file" onChange={handleFileChange} />
      
      {/* Display image preview */}
      {imagePreviewUrl && <img src={imagePreviewUrl} alt="Uploaded Image" style={{ maxWidth: '300px', marginTop: '20px' }} />}
      
      <button onClick={handleSubmit}>Predict</button>

      {/* Display prediction result */}
      {prediction && <h3>Prediction: {prediction}</h3>}
      {confidence && <h4>Confidence: {confidence.toFixed(2)}%</h4>}

      {/* Display error message */}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

export default Prediction;
