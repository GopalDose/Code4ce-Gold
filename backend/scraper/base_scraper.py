# backend/scraping/base_scraper.py

import requests
from bs4 import BeautifulSoup
import time
import random

class BaseScraper:
    def __init__(self, base_url):
        self.base_url = base_url

    def fetch_page(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_html(self, html_content):
        return BeautifulSoup(html_content, 'lxml')

    def wait(self):
        # Random wait to avoid getting blocked
        time.sleep(random.uniform(1, 3))
