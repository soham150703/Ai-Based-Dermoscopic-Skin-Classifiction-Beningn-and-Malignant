import React, { useEffect, useState } from "react";
import axios from "axios";
import "./Database.css"; // Custom CSS file for styling

export default function Database2() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const getProducts = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:3001/database2");
        setData(response.data.data);
        console.log(response.data.data);
      } catch (error) {
        console.error("Error fetching products:", error);
      }
    };
    getProducts();
  }, []);
  
  return (
    <div className="database-container">
      <h2>Product Database</h2>
      <div className="product-grid">
        {data.map((item, index) => (
          <div key={index} className="product-card">
            <img
              src={item.image}
              alt={item.CorrectLabel}
              className="product-image"
            />
            <div className="product-label">Correct Label: {item.CorrectLabel}</div>
          </div>
        ))}
      </div>
    </div>
  );
}