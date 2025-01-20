import React, { useEffect, useState, useContext } from 'react';
import './LatestNews.css';
import LatestCard from '../LatestCard/LatestCard';
import SentimentDashboard from '../SentimentPieCharts/SentimentPieCharts';
import { AuthContext } from '../../context/AuthContext';

const LatestNews = () => {
    const { category, country } = useContext(AuthContext);

    const [articles, setArticles] = useState([]);
    const [latestNew, setLatestNew] = useState(true);
    const apiKey = '';

    useEffect(() => {
        // Fetch data from the API
        const fetchArticles = async () => {
            try {
                if (category === 'All' || country === 'Other') {
                    // Local API
                    const response = await fetch('http://127.0.0.1:5000/api/articles');
                    const data = await response.json();
                    setArticles(data.processed_articles);
                } else {
                    // NewsAPI
                    const query = `${country} ${category !== 'All' ? category : 'defence'}`;
                    const tempData = await fetch(`https://newsapi.org/v2/everything?q=${query}&pageSize=20&apiKey=${apiKey}`);
                    const newsData = await tempData.json();  // Await the JSON conversion
                    setArticles(newsData.articles);  // Set articles from NewsAPI
                }
            } catch (error) {
                console.error("Error fetching articles:", error);
            }
        };

        fetchArticles();  // Call the async function
    }, [country, category]);  // Dependency on category and country

    return (
        <div className="latest">
            <div className="head">
                <h2
                    onClick={() => setLatestNew(true)}
                    className={latestNew === true ? 'active' : ''}>Latest News</h2>
                <h2
                    onClick={() => setLatestNew(false)}
                    className={latestNew === false ? 'active' : ''}>Dashboard</h2>
            </div>
            <div className="card-container"> 
                {
                    latestNew === true && <div className="article">
                        {articles.map((article, index) => (
                            <LatestCard
                                key={index}
                                title={article.article?.title || article.title}  // Handling both cases (local API and NewsAPI)
                                description={article.article?.summary || article.description}  // Fallback to NewsAPI description
                                imageUrl={article.article?.image ? `https://www.aljazeera.com${article.article.image}` : article.urlToImage}  // Handle imageUrl correctly
                                url={article.article?.url || article.url}  // Fallback to correct URL
                                article={article}
                            />
                        ))}
                    </div>
                }
                {
                    latestNew === false &&
                    <>
                        <SentimentDashboard />
                    </>
                }
            </div>
        </div>
    );
};

export default LatestNews;
