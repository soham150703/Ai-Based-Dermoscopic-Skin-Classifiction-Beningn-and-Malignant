import React, { useState } from "react";
import axios from "axios";
export default function Form() {
  const [label, setLabel] = useState("");
  const [img, setImg] = useState("");
  function convert(file) {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.readAsDataURL(file);
      fileReader.onload = () => {
        resolve(fileReader.result);
      };
      fileReader.onerror = (error) => {
        reject(error);
      };
    });
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    const base64 = await convert(file);
    setImg(base64);
  };
const Submit= async(e)=>{
    e.preventDefault();
    try {
        const response = await axios.post(
          "http://127.0.0.1:3001/database2",
          { img , label },
          
        
        );
        window.location.reload();
        
      } catch (error) {
        console.log(error.message);
      }
 }
  return (
    <div>
      <form onSubmit={Submit}>
        <label>Upload Image</label>
        <input
          type="file"
          name="image"
          accept="image/jpeg, image/png"
          onChange={handleFileUpload}
          required
        />
        <br />
        <label>Enter correct label : </label>
        <input
          type="text"
          required
          onChange={(e) => {
            setLabel(e.target.value);
          }}
        />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
}