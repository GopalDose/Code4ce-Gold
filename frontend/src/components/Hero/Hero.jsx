import React, { useContext } from 'react';
import './Hero.css';
import SearchBar from '../SearchBar/SearchBar';
import { AuthContext } from '../../context/AuthContext';

// Import your images
// import indiaImage from '../../assets/India.jpg';
import usImage from '../../assets/us.jpg';
import indiaImage from '../../assets/india.jpg';
import ukImage from '../../assets/uk.jpg';
import europeImage from '../../assets/europe.jpg';

const Hero = () => {
    const { category, setCategory, country, setCountry } = useContext(AuthContext);

    // Function to handle category click
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

    // Determine background image based on the selected country
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
                return usImage; // You can set a default image if needed
        }
    })();

    return (
        <div className="Hero" style={{ backgroundImage: `url(${backgroundImage})` }}>
            <div className="overlay">
                <h2 className='country'>{country}</h2>
                <SearchBar />
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
                <div className="country">
                    <select name="country" id="country" value={country} onChange={handleCountryChange}>
                        <option value="">Select Country</option>
                        <option value="India">India</option>
                        <option value="US">US</option>
                        <option value="UK">UK</option>
                        <option value="Europe">Europe</option>
                    </select>
                </div>
            </div>
        </div>
    );
};

export default Hero;
