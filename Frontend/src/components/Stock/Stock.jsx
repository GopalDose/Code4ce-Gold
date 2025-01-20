import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './Stock.css';

const Stock = () => {
    const [latestNewsData, setLatestNewsData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchNewsData = async () => {
            try {
                const apiKey = "";
                const response = await axios.get(`https://finnhub.io/api/v1/news?category=general&token=${apiKey}`);

                // Limit the news to the first 20 items
                const limitedNewsData = response.data.slice(0, 20);
                setLatestNewsData(limitedNewsData);
                console.log(limitedNewsData)
            } catch (error) {
                setError('Failed to fetch news data.');
            } finally {
                setLoading(false); 
            }
        };

        fetchNewsData();
    }, []);

    if (loading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>{error}</div>;
    }

    return (
        <div className="stock">
            <h2>General</h2>
            <div className="stocknews">
                {latestNewsData.map((newsItem, index) => (
                    <div key={index} className="card">
                        <div className="image">
                            <img src={newsItem.image} alt="Stock News" />
                        </div>
                        <div className="content">
                            {newsItem.headline}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Stock;
