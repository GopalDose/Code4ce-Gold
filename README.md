# 🌐 WarCast – AI-based Defense News Aggregator

WarCast is an AI-powered, multilingual news aggregator built to streamline and personalize defense-related news. It collects real-time articles from over 10 trusted sources, analyzes their sentiment, and delivers short, focused summaries to help users stay informed without information overload.

---

## 🚀 Features

- 🔍 **Multilingual News Scraper**: Aggregates real-time news from 10+ global defense sources.
- 🧠 **Personalized Sentiment Feed**: Uses DistilBERT for fine-tuned sentiment classification.
- ✂️ **AI Summarization**: Generates crisp, 150-word summaries using BART-based transformer models.
- 🌍 **Multilingual Support**: Enables access to news in multiple languages.
- 📰 **Categorized Feed**: Classifies content under defense, geopolitics, cyberwarfare, and more.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **AI Models**: 
  - [DistilBERT](https://huggingface.co/distilbert-base-uncased) for sentiment analysis  
  - [BART](https://huggingface.co/facebook/bart-large-cnn) for summarization
- **Scraping**: `BeautifulSoup`, `Newspaper3k`, `Requests`
- **Deployment**: Flask REST APIs with frontend-ready JSON responses

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/WarCast.git
cd WarCast
pip install -r requirements.txt
python app.py
