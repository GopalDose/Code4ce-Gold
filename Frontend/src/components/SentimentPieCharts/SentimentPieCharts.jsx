import React, { useEffect, useState } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import './SentimentPieCharts.css';

ChartJS.register(ArcElement, Tooltip, Legend);

const chartOptions = {
  plugins: {
    legend: {
      position: 'bottom',
      labels: {
        usePointStyle: true,
        padding: 20,
        font: {
          size: 12
        }
      }
    },
    tooltip: {
      callbacks: {
        label: (context) => `${context.label}: ${context.raw}%`
      }
    }
  },
  cutout: '65%',
  responsive: true,
  maintainAspectRatio: false,
  animation: {
    animateScale: true,
    animateRotate: true
  }
};

// Function to create chart data based on sentiment values
const createChartData = (source) => ({
  labels: ['Positive', 'Neutral', 'Negative'],
  datasets: [{
    data: [source.positive, source.neutral, source.negative],
    backgroundColor: [
      'rgba(52, 211, 153, 0.8)',  // Positive - Green
      'rgba(251, 191, 36, 0.8)',  // Neutral - Yellow
      'rgba(239, 68, 68, 0.8)'    // Negative - Red
    ],
    borderColor: [
      'rgba(52, 211, 153, 1)',
      'rgba(251, 191, 36, 1)',
      'rgba(239, 68, 68, 1)'
    ],
    borderWidth: 2,
    hoverOffset: 4
  }]
});

const SentimentChart = ({ source }) => (
  <div className="chart-card">
    <h3 className="chart-title">{source.name}</h3>
    <div className="chart-container">
      <Doughnut data={createChartData(source)} options={chartOptions} />
    </div>
  </div>
);

const SourceSection = ({ title, sources }) => (
  <div className="source-section">
    <h2 className="section-title">{title}</h2>
    <div className="charts-grid">
      {sources.map((source, index) => (
        <SentimentChart key={`${source.name}-${index}`} source={source} />
      ))}
    </div>
  </div>
);

const SentimentDashboard = () => {
  const [articles, setArticles] = useState([]);
  const [averageSentiment, setAverageSentiment] = useState({
    positive: 0,
    neutral: 0,
    negative: 0
  });

  // Fetch articles from the API
  useEffect(() => {
    const fetchArticles = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/api/articles'); // Replace with your actual API URL
        const data = await response.json();
        setArticles(data.processed_articles); // Assuming the API response has a 'processed_articles' array

        // Filter articles based on URL containing "aljazeera.com"
        const alJazeeraArticles = data.processed_articles.filter(article =>
          article.article.url && article.article.url.includes('aljazeera.com') // Check if URL contains 'aljazeera.com'
        );

        if (alJazeeraArticles.length > 0) {
          const totalSentiment = alJazeeraArticles.reduce(
            (acc, article) => {
              if (article.article.sentiment) { // Ensure article.article.sentiment exists
                acc.positive += article.article.sentiment.details.pos;
                acc.neutral += article.article.sentiment.details.neu;
                acc.negative += article.article.sentiment.details.neg;
              }
              return acc;
            },
            { positive: 0, neutral: 0, negative: 0 }
          );

          const avgSentiment = {
            positive: totalSentiment.positive / alJazeeraArticles.length,
            neutral: totalSentiment.neutral / alJazeeraArticles.length,
            negative: totalSentiment.negative / alJazeeraArticles.length
          };

          setAverageSentiment(avgSentiment);
        }
      } catch (error) {
        console.error('Error fetching articles:', error);
      }
    };


    fetchArticles();
  }, []);

  return (
    <div className="sentiment-container">
      <div className="dashboard">
        <h1 className="dashboard-title">Defense News Sentiment Analysis</h1>

        <SourceSection title="Sources" sources={[
          { name: 'Al Jazeera', ...averageSentiment },
          {
            name: 'BBC',
            positive: 30,
            neutral: 45,
            negative: 25
          },
          {
            name: 'Reuters',
            positive: 35,
            neutral: 45,
            negative: 20
          },
          {
            name: 'The Guardian',
            positive: 28,
            neutral: 47,
            negative: 25
          },
          {
            name: 'Al Jazeera',
            positive: 25,
            neutral: 45,
            negative: 30
          }]} />
        {/* You can include more sections for other sources like domestic, etc. */}
      </div>
    </div>
  );
};

export default SentimentDashboard;
