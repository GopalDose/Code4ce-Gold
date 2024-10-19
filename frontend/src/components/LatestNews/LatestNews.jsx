import React, { useEffect, useState, useContext } from 'react';
import './LatestNews.css';
import LatestCard from '../LatestCard/LatestCard';
import { AuthContext } from '../../context/AuthContext'; 
import SentimentPieCharts from '../SentimentPieCharts/SentimentPieCharts';
import MultilingualSentimentAnalysis from '../MultilingualSentimentAnalysis/MultilingualSentimentAnalysis';

const LatestNews = ({ homeState, setHomeState }) => {
  const [latestNewsData, setLatestNewsData] = useState([]);
  const { category, language, country, isLoggedIn } = useContext(AuthContext);  // Get category, language, and country from context

  useEffect(() => {
    const fetchData = async () => {
      try {
        let response;
        const apiKey = 'e2ab1bb006e64778a09cc161ce78da85'; // Replace with your News API key
        
        if (category === 'All') {
          // Fetch all articles from backend API
          response = await fetch('http://127.0.0.1:5000/api/articles');
        } else {
          // Concatenate 'defence' with country value for the News API query if category is 'All', else concatenate the country with the category
          const query = `${country} ${category !== 'All' ? category : 'defence'}`;
          
          // Fetch from News API based on category, language, and country with appropriate concatenation
          response = await fetch(`https://newsapi.org/v2/everything?q=${query}&language=${language}&from=2024-09-19&sortBy=publishedAt&pageSize=20&apiKey=${apiKey}`);
        }

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Adjust the data structure if needed
        if (category === 'All') {
          setLatestNewsData(data); // assuming data is already in correct format
        } else {
          setLatestNewsData(data.articles); // Use articles from News API response
        }
      } catch (error) {
        console.error("Error fetching articles:", error);
      }
    };

    fetchData();
  }, [category, language, country]); // Refetch when category, language, or country changes

  return (
    <div className="latest">
      <div className="head">
        <h2
          onClick={() => setHomeState('Latest')}
          className={homeState === 'Latest' ? 'active' : ''}
        >
          Latest News
        </h2>
        <h2
          onClick={() => setHomeState('Reco')}
          className={homeState === 'Reco' ? 'active' : ''}
        >
          Dashboard
        </h2>
        {
          isLoggedIn && <h2
            onClick={() => setHomeState('sub')}
            className={homeState === 'sub' ? 'active' : ''}
          >
            For You
          </h2>
        }
      </div>
      <div className="card-container">
        {homeState === 'Latest' &&
          latestNewsData.map((newsItem, index) => (
            <LatestCard
              key={index}
              title={newsItem.title}
              description={newsItem.summary || newsItem.description} // Use summary if available
              imageUrl={newsItem.image || newsItem.urlToImage || "https://salonlfc.com/wp-content/uploads/2018/01/image-not-found-1-scaled.png"} // Use image if available
              newsItem={newsItem}
            />
          ))}

        {homeState === "Reco" && (
          <>
            <SentimentPieCharts />
            <MultilingualSentimentAnalysis />
          </>
        )}

        {homeState === "sub" &&<>
        <div className="foryou">
          <div className="head">
            Bookmark
          </div>
          <LatestCard
              title="Newspaper offices hit by gunfire in Mexico’s Sinaloa state capital"
              description="Gunmen shoot at office building of respected Mexican newspaper in Sinaloa capital Culiacan." // Use summary if available
              imageUrl="https://www.aljazeera.com/wp-content/uploads/2024/10/2019-10-18T000000Z_680659756_RC13B02793C0_RTRMADP_3_MEXICO-VIOLENCE-SINALOA-1-1729295013.jpg?resize=770%2C513&quality=80" // Use image if available
              newsItem=""
            />
          <div className="head">
            Liked
          </div>
          <LatestCard
              title="Newspaper offices hit by gunfire in Mexico’s Sinaloa state capital"
              description="Gunmen shoot at office building of respected Mexican newspaper in Sinaloa capital Culiacan." // Use summary if available
              imageUrl="https://www.aljazeera.com/wp-content/uploads/2024/10/2019-10-18T000000Z_680659756_RC13B02793C0_RTRMADP_3_MEXICO-VIOLENCE-SINALOA-1-1729295013.jpg?resize=770%2C513&quality=80" // Use image if available
              newsItem=""
            />
        </div>
        </>} {/* Display "Gopal" for the For You section */}
      </div>
    </div>
  );
};

export default LatestNews;
