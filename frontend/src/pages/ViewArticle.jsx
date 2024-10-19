import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const ViewArticle = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { title, description, imageUrl, newsItem } = location.state || {};

  // State to manage active icons
  const [isLiked, setIsLiked] = useState(false);
  const [isBookmarked, setIsBookmarked] = useState(false);

  // Helper function to truncate content only if length is greater than 200 and less than 250 characters
  const truncateContent = (content) => {
    if (content) {
      const contentLength = content.length;
      // Truncate content if it is between 200 and 250 characters
      if (contentLength > 200 && contentLength < 250) {
        return `${content.substring(0, 200)}...`;
      }
      return content; // Return full content otherwise
    }
    return "Content not available.";
  };

  return (
    <div className="viewarti">
      <button onClick={() => navigate(-1)} className="back-button"><i className='bx bx-arrow-back'></i> Back</button>
      <h2>{newsItem.title}</h2>
      <div className="summary">
        {newsItem.summary || newsItem.description}
      </div>
      <img src={newsItem.image || newsItem.urlToImage || "https://salonlfc.com/wp-content/uploads/2018/01/image-not-found-1-scaled.png"} alt={title} />
      <p>
        {truncateContent(newsItem.content || newsItem.full_content)}
      </p>
      <div className="action">
        <i 
          className={`bx bxs-heart ${isLiked ? 'active' : ''}`} 
          onClick={() => setIsLiked(!isLiked)} 
        />
        <i 
          className={`bx bxs-bookmark-star ${isBookmarked ? 'active' : ''}`} 
          onClick={() => setIsBookmarked(!isBookmarked)} 
        />
        <i class='bx bxs-share-alt' ></i>
      </div>
    </div>
  );
};

export default ViewArticle;
