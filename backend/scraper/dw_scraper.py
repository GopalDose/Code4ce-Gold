from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
class DeutscheWelleScraper:
    base_url = "https://www.dw.com/en"

    def get_latest_articles(self, fetch_url):
        url = f"{self.base_url}/"
        content = fetch_url(url)
        if content is None:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        articles = []

        for item in soup.find_all('article'):
            title = item.find('h2').get_text(strip=True) if item.find('h2') else 'No title'
            link = item.find('a')['href'] if item.find('a') else None
            summary = item.find('p').get_text(strip=True) if item.find('p') else 'No summary'

            if link:
                articles.append({
                    'title': title,
                    'link': f"{self.base_url}{link}",
                    'summary': summary,
                    'published_date': None
                })

        return articles