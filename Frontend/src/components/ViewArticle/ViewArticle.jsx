import React from 'react';
import './ViewArticle.css';
import { useLocation, useNavigate } from 'react-router-dom';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

// Register the necessary components for Chart.js
ChartJS.register(ArcElement, Tooltip, Legend); 

const ViewArticle = () => {
    const location = useLocation();
    const { article } = location.state || {};
    const navigate = useNavigate();

    // Safe checks for sentiment data
    const sentimentData = article?.article?.sentiment?.details ? {
        labels: ['Negative', 'Neutral', 'Positive'],
        datasets: [
            {
                label: 'Sentiment',
                data: [
                    article.article.sentiment.details.neg || 0,
                    article.article.sentiment.details.neu || 0,
                    article.article.sentiment.details.pos || 0,
                ],
                backgroundColor: ['#FF5733', '#C2C2C2', '#4CAF50'], // Colors for negative, neutral, and positive
                borderColor: ['#FF5733', '#C2C2C2', '#4CAF50'],
                borderWidth: 1,
            },
        ],
    } : null; // Avoid showing the pie chart if sentiment details are not available

    const options = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
        },
    };

    return (
        <section>
            <button onClick={() => navigate(-1)} className="back-button"><i className='bx bx-arrow-back'></i> Back</button>
            <div className="title">
                {article?.article?.title || article?.title || 'Untitled Article'}
            </div>
            <div className="desc">
                {article?.article?.desc || article?.description || 'No description available.'}
            </div>
            <div className="img">
                <img 
                    src={article?.article?.image ? `https://www.aljazeera.com${article.article.image}` : article?.urlToImage || '/fallback-image.jpg'} 
                    alt="Article" 
                />
            </div>
            <div className="summary">
                {article.article?.summary || article.content}
            </div>
            <a href={article?.article?.url || article?.url} className="view-btn" target="_blank" rel="noopener noreferrer">View On Source</a>
            {sentimentData && (
                <div className="senti">
                    <div className="pie-chart">
                        <Pie data={sentimentData} options={options} />
                    </div>
                </div>
            )}
        </section>
    );
};

export default ViewArticle;
