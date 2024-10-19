import React, { useState } from 'react';
import './App.css';
import Header from './components/Header/Header';
import Footer from './components/Footer/Footer';
import Home from './pages/Home';
import UpperHeader from './components/UpperHeader/UpperHeader';
import LoginPopup from './components/LoginPopup/LoginPopup';
import { AuthProvider } from './context/AuthContext';
import ViewArticle from './pages/ViewArticle';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'; // Import necessary components
import SearchResult from './pages/SearchResult';

const App = () => {
  const [showLogin, setShowLogin] = useState(false);
  
  return (
    <AuthProvider>
      {showLogin && <LoginPopup setShowLogin={setShowLogin} />}
      <div className="container">
        <UpperHeader />
        <Header setShowLogin={setShowLogin} showLogin={showLogin} />
        
        {/* Wrap your routes with Router and define them here */}
        <Router>
          <Routes>
            <Route path="/" element={<Home />} />          {/* Home route */}
            <Route path="/view-article" element={<ViewArticle />} /> {/* View Article route */}
            <Route path="/searchresult" element={<SearchResult />} />
          </Routes>
        </Router>

        <Footer />
      </div>
    </AuthProvider>
  );
}

export default App;
