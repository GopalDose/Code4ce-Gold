# busiStand_scraper.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

class BusinessStandardScraper:
    base_url = "https://www.business-standard.com"

    def get_latest_articles(self, fetch_url=None):
        url = f"{self.base_url}/category/current-affairs/defence"
        # Set up Selenium with ChromeDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode (no browser window)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)

        # Wait for a few seconds to allow the page to load
        time.sleep(5)

        # Get page source and close the browser
        content = driver.page_source
        driver.quit()

        if not content:
            return []

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        articles = []

        # Example logic for parsing articles
        for item in soup.find_all('div', class_='listing-txt'):
            title = item.find('a').get_text(strip=True) if item.find('a') else 'No title'
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
