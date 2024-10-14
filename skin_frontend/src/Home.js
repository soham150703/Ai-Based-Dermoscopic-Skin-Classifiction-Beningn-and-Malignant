// src/Home.js
import React from 'react';

function Home() {
  return (
    <div>
      <h2>What is Dermoscopy?</h2>
      <p>
        Dermoscopy is a non-invasive diagnostic method used to examine skin lesions with the help of a dermatoscope.
        It helps in identifying different types of skin conditions, including melanoma (skin cancer).
      </p>

      <h2>How is it helpful?</h2>
      <p>
        Dermoscopy aids in early detection of malignant melanoma by allowing doctors to visualize subsurface skin structures
        that are not visible to the naked eye. This method improves diagnostic accuracy and enables better treatment outcomes.
      </p>

      <h2>Benign vs Malignant Cells</h2>
      <p>
        Benign cells are non-cancerous and don't spread to other parts of the body, while malignant cells are cancerous, 
        aggressive, and capable of spreading (metastasis).
      </p>

      <div className="images">
        <div className="image-item">
          <img src={process.env.PUBLIC_URL + '/images/dermo.jpg'} alt="Dermoscopy" />
          <p>Dermoscopy Example</p>
        </div>
        <div className="image-item">
          <img src={process.env.PUBLIC_URL + '/images/benign.jpg'} alt="Benign vs Malignant" />
          <p>Benign vs Malignant Cells</p>
        </div>
      </div>
    </div>
  );
}

export default Home;
