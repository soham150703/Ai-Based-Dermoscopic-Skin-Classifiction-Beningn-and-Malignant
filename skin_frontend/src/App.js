import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Home from './Home';
import Prediction from './Prediction';
import Prediction2 from './Prediction2';
import About from './About';
import Contact from './Contact';
import Database from './Database';
import Database2 from './Database2'; // Import the new Database component
import './App.css';
import Form from './form';
import Form2 from './form2';

function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  const toggleMenu = () => {
    console.log('Menu toggled:', !menuOpen); // Debugging log
    setMenuOpen(!menuOpen);
  };

  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Dermoscopy: Classification of Benign and Malignant Cells</h1>
          {/* Burger Icon */}
          <div className="burger-icon" onClick={toggleMenu}>
            <span></span>
            <span></span>
            <span></span>
          </div>
          {/* Navigation Menu */}
          <nav className={`nav-menu ${menuOpen ? 'open' : ''}`}>
            <ul>
              <li>
                <Link to="/" onClick={() => setMenuOpen(false)}>
                  Home
                </Link>
              </li>
              <li>
                <Link to="/prediction" onClick={() => setMenuOpen(false)}>
                  Prediction
                </Link>
              </li>
              <li>
                <Link to="/prediction2" onClick={() => setMenuOpen(false)}>
                  Prediction2
                </Link>
              </li>
              <li>
                <Link to="/about" onClick={() => setMenuOpen(false)}>
                  About
                </Link>
              </li>
              <li>
                <Link to="/contact" onClick={() => setMenuOpen(false)}>
                  Contact
                </Link>
              </li>
              <li>
                <Link to="/database" onClick={() => setMenuOpen(false)}>
                  Database
                </Link>
              </li>
            </ul>
          </nav>
        </header>

        {/* Main Content */}
        <div className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/prediction" element={<Prediction />} />
            <Route path="/prediction2" element={<Prediction2 />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/form" element={<Form />} />
            <Route path="/form2" element={<Form2 />} />
            <Route path="/database" element={<Database />} />
            <Route path="/database2" element={<Database2 />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
