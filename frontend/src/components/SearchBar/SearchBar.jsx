import React, { useState, useContext } from 'react';
import './SearchBar.css';
import logo from '../../assets/logo.png';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import { AuthContext } from '../../context/AuthContext';

const SearchBar = () => {
  const [query, setQuery] = useState('');
  const { language } = useContext(AuthContext);
  const apiKey = 'e2ab1bb006e64778a09cc161ce78da85'; // Your News API key
  const navigate = useNavigate(); // Initialize navigate

  const handleSearch = async (e) => {
    e.preventDefault();

    if (!query) return;

    try {
      const response = await fetch(`https://newsapi.org/v2/everything?q=${query}&language=${language}&from=2024-09-19&sortBy=publishedAt&pageSize=20&apiKey=${apiKey}`);
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      // Navigate to SearchResult page and pass the data
      navigate('/searchresult', { state: { articles: data.articles } });
    } catch (error) {
      console.error("Error fetching articles:", error);
    }
  };

  return (
    <div className="search">
      <div className="searchbar">
        <img src={logo} alt="Logo" />
        <form onSubmit={handleSearch}>
          <input 
            type="text" 
            placeholder='Search here' 
            value={query} 
            onChange={(e) => setQuery(e.target.value)}
          />
          <label htmlFor="btn"><i className='bx bx-search-alt-2'></i></label>
          <input type="submit" id='btn' hidden />
        </form>
      </div>
    </div>
  );
}

export default SearchBar;