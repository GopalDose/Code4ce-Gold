import React, { useContext, useState } from 'react'
import './Hero.css'
import usImage from '../../assets/us.jpg';
import indiaImage from '../../assets/india.jpg';
import ukImage from '../../assets/uk.jpg';
import europeImage from '../../assets/europe.jpg';
import logo from '../../assets/logo.png';
import { AuthContext } from '../../context/AuthContext';
import { useNavigate } from 'react-router-dom';

const Hero = () => {
    const [query, setQuery] = useState('');
    const apiKey = 'e2ab1bb006e64778a09cc161ce78da85';
    const navigate = useNavigate();
    const { category, setCategory, country, setCountry } = useContext(AuthContext);

    const handleSearch = async (e) => {
        e.preventDefault();

        if (!query) return;

        try {
            // Declare response with 'let'
            let response = await fetch(`https://newsapi.org/v2/everything?q=${query}&pageSize=20&apiKey=${apiKey}`);

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            navigate('/searchresult', { state: { articles: data.articles } });
        } catch (error) {
            console.error("Error fetching articles:", error);
        }
    };

    const handleCategoryClick = (categoryName) => {
        if (category === categoryName) {
            setCategory('All'); // Reset to 'All' if the same category is clicked
        } else {
            setCategory(categoryName); // Set the new category
        }
    };

    const handleCountryChange = (event) => {
        setCountry(event.target.value); // Update country state with selected value
    };

    const backgroundImage = (() => {
        switch (country) {
            case 'India':
                return indiaImage;
            case 'US':
                return usImage;
            case 'UK':
                return ukImage;
            case 'Europe':
                return europeImage;
            default:
                return usImage; // Default background image
        }
    })();

    return (
        <div className="hero" style={{ backgroundImage: `url(${backgroundImage})` }}>
            <div className="overlay">
                <div className="country">
                    {country}
                </div>
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
                <div className="subCategory">
                    <ul>
                        <li>
                            <a href="#" onClick={() => handleCategoryClick('Airforce')}>Airforce</a>
                        </li>
                        <li>
                            <a href="#" onClick={() => handleCategoryClick('Navy')}>Navy</a>
                        </li>
                        <li>
                            <a href="#" onClick={() => handleCategoryClick('Terrorism')}>Terrorism</a>
                        </li>
                        <li>
                            <a href="#" onClick={() => handleCategoryClick('Cyber Crimes')}>Cyber Crime</a>
                        </li>
                        <li>
                            <a href="#" onClick={() => handleCategoryClick('Politics')}>Politics</a>
                        </li>
                    </ul>
                </div>
                <select value={country} onChange={handleCountryChange}>
                    <option value="Other">Any</option>
                    <option value="India">India</option>
                    <option value="US">United States</option>
                    <option value="Europe">Europe</option>
                    <option value="UK">United Kingdom</option>
                </select>
            </div>
        </div>
    );
}

export default Hero;
