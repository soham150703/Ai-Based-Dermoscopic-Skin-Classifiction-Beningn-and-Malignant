import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Home from './Home';
import Prediction from './Prediction';
import './App.css';

function App() {
  const [selectedPage, setSelectedPage] = useState('');

  const handleNavigation = (event) => {
    setSelectedPage(event.target.value);
  };

  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Dermoscopy: Classification of Benign and Malignant Cells</h1>
          <select value={selectedPage} onChange={handleNavigation}>
            <option value="">-- Select Page --</option>
            <option value="home">Home</option>
            <option value="prediction">Prediction</option>
          </select>
          <div className="nav-links">
            {selectedPage === 'home' && <Link to="/">Go to Home</Link>}
            {selectedPage === 'prediction' && <Link to="/prediction">Go to Prediction</Link>}
          </div>
        </header>

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/prediction" element={<Prediction />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
