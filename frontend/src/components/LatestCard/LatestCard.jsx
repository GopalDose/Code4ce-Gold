import React from 'react';
import './LatestCard.css';
import { Link } from 'react-router-dom'; // Import Link

const LatestCard = ({ title, description, imageUrl, newsItem }) => {
  const defaultImage = "https://salonlfc.com/wp-content/uploads/2018/01/image-not-found-1-scaled.png"; // Default image URL

  return (
    <Link to="/view-article" state={{ title, description, imageUrl, newsItem }} className="card">
    <div className="card">
      <div className="latestcard">
        <span>2 hours ago</span>
        <h3>{title}</h3>
        <p className='description '>{description}</p>
        <div className="category">
          <ul>
            <li>News</li>
            <li>Updates</li>
          </ul>
        </div>
      </div>
      <img onError={(e) => { e.target.onerror = null; e.target.src = defaultImage; }} // Set default image on error
       src={imageUrl} alt="News" />
    </div>
    </Link>
  );
};

export default LatestCard;
