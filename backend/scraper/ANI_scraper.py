import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

class ANIScraper:
    base_url = "https://www.aninews.in"

    def get_latest_articles(self, fetch_url=None):
        url = f"{self.base_url}/category/national"
        # Set up Selenium with ChromeDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode (no browser window)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)

        # Wait for a few seconds to allow the page to load
        time.sleep(5)

        # Get the page source and close the browser
        content = driver.page_source
        driver.quit()

        if not content:
            return []

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        articles = []

        # Example logic for parsing articles
        for item in soup.find_all('article'):
            title = item.find('h2').get_text() if item.find('h2') else 'No title'
            link = item.find('a')['href'] if item.find('a') else None
            summary = item.find('p').get_text() if item.find('p') else 'No summary'
            content = item.get_text(strip=True) if item else 'No content'
            category = item.find('div', class_='post-category').get_text() if item.find('div', class_='post-category') else 'No category'
            image = item.find('img')['src'] if item.find('img') else 'No image'

            if link:
                articles.append({
                    'title': title,
                    'link': f"{self.base_url}{link}",
                    'summary': summary,
                    'content': content,
                    'category': category,
                    'image': image,
                    'published_date': None
                })

        return articles

    def save_to_csv(self, articles, filename='ani_articles.csv'):
        # Define the headers for the CSV file
        headers = ['Title', 'Link', 'Summary', 'Content', 'Category', 'Image', 'Published Date']

        # Write data to the CSV file
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for article in articles:
                writer.writerow({
                    'Title': article['title'],
                    'Link': article['link'],
                    'Summary': article['summary'],
                    'Content': article['content'],
                    'Category': article['category'],
                    'Image': article['image'],
                    'Published Date': article['published_date']
                })

# Usage
scraper = ANIScraper()
articles = scraper.get_latest_articles()
scraper.save_to_csv(articles)
