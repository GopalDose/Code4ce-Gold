import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
// import './SentimentPieCharts.css';

ChartJS.register(ArcElement, Tooltip, Legend);

const sources = {
  domestic: [
    {
      name: 'Le Monde',
      region: 'France',
      language: 'French',
      positive: 30,
      neutral: 48,
      negative: 22
    },
    {
      name: 'El País',
      region: 'Spain',
      language: 'Spanish',
      positive: 35,
      neutral: 45,
      negative: 20
    },
    {
      name: 'Der Spiegel',
      region: 'Germany',
      language: 'German',
      positive: 25,
      neutral: 50,
      negative: 25
    }
  ],
  international: [
    {
      name: 'Xinhua',
      region: 'China',
      language: 'Chinese',
      positive: 45,
      neutral: 40,
      negative: 15
    },
    {
      name: 'Al Jazeera Arabic',
      region: 'Middle East',
      language: 'Arabic',
      positive: 28,
      neutral: 42,
      negative: 30
    },
    {
      name: 'Russia Today',
      region: 'Russia',
      language: 'Russian',
      positive: 42,
      neutral: 38,
      negative: 20
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
    <p className="chart-subtitle">
      {source.region} ({source.language})
    </p>
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

const calculateAverages = (sourceGroup) => {
  const totals = sourceGroup.reduce((acc, source) => ({
    positive: acc.positive + source.positive,
    neutral: acc.neutral + source.neutral,
    negative: acc.negative + source.negative
  }), { positive: 0, neutral: 0, negative: 0 });

  const count = sourceGroup.length;
  return {
    positive: Math.round(totals.positive / count),
    neutral: Math.round(totals.neutral / count),
    negative: Math.round(totals.negative / count)
  };
};

const SentimentDashboard = () => {
  const domesticAvg = calculateAverages(sources.domestic);
  const internationalAvg = calculateAverages(sources.international);

  return (
    <div className="sentiment-container">
      <div className="dashboard">
        <h1 className="dashboard-title">Global Defense News Sentiment Analysis</h1>
        
        <SourceSection title="European Sources" sources={sources.domestic} />
        <SourceSection title="International Sources" sources={sources.international} />
      </div>
    </div>
  );
};

export default SentimentDashboard;