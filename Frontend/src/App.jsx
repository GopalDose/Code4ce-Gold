import React from 'react'
import './App.css'
import Navbar from './components/Navbar/Navbar'
import Home from './components/Home/Home'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'; // Import necessary components
import ViewArticle from './components/ViewArticle/ViewArticle'
import { AuthProvider } from './context/AuthContext';
import SearchResult from './components/SearchResult/SearchResult';

const App = () => {
  return (
    <AuthProvider>
      <div className="container">
        <Navbar />

        <Router>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/view-article" element={<ViewArticle />} />
            <Route path="/searchresult" element={<SearchResult />} />
          </Routes>
        </Router>
      </div>
    </AuthProvider>
  )
}

export default App