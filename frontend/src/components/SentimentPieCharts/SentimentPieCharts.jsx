import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import './SentimentPieCharts.css';

ChartJS.register(ArcElement, Tooltip, Legend);

const sources = {
  domestic: [
    {
      name: 'The Hindu',
      positive: 35,
      neutral: 45,
      negative: 20
    },
    {
      name: 'Times of India',
      positive: 40,
      neutral: 40,
      negative: 20
    },
    {
      name: 'Hindustan Times',
      positive: 38,
      neutral: 42,
      negative: 20
    },
    {
      name: 'NDTV',
      positive: 32,
      neutral: 43,
      negative: 25
    }
  ],
  international: [
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
    }
  ]
};

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
  return (
    <div className="sentiment-container">
      <div className="dashboard">
        <h1 className="dashboard-title">Defense News Sentiment Analysis</h1>
        
        <SourceSection title="Domestic Sources" sources={sources.domestic} />
        <SourceSection title="International Sources" sources={sources.international} />
      </div>
    </div>
  );
};

export default SentimentDashboard;