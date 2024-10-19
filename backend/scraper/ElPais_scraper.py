from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import concurrent.futures
import csv

class ElPaisEnglishScraper:
    base_url = "https://english.elpais.com/buscador/defence/"

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
            image = item.find('img')['src'] if item.find('img') else 'No image'

            if link:
                articles.append({
                    'title': title,
                    'link': f"{self.base_url}{link}",
                    'summary': summary,
                    'published_date': None,
                    'image': image
                })

        return articles

    def scrape_article_content(self, article, fetch_url):
        """Visit the article URL and scrape the full content."""
        content = fetch_url(article['link'])
        if content is None:
            article['full_content'] = 'No content'
            return article

        soup = BeautifulSoup(content, 'html.parser')
        full_content = soup.find('div', class_='article-body').get_text(strip=True) if soup.find('div', class_='article-body') else 'No content'
        article['full_content'] = full_content

        # If needed, you can also extract the published date here
        published_date = soup.find('time')['datetime'] if soup.find('time') else article['published_date']
        article['published_date'] = published_date

        return article

    def save_to_csv(self, articles, filename='articles.csv'):
        """Save the list of articles to a CSV file."""
        with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'link', 'summary', 'published_date', 'image', 'full_content']  # Add 'full_content'
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()  # Write the header
            for article in articles:
                writer.writerow(article)

        print(f"Saved {len(articles)} articles to {filename}.")

# Example usage
def fetch_url(url):
    # Set up the Selenium Chrome driver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run headlessly
    options.add_argument('--no-sandbox')  # Bypass OS security model
    options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)

    try:
        # Wait for the page to load the necessary elements
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'article'))
        )
        content = driver.page_source
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        content = None
    finally:
        driver.quit()

    return content

scraper = ElPaisEnglishScraper()
articles = scraper.get_latest_articles(fetch_url)

# Scrape full content for each article concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    articles = list(executor.map(lambda article: scraper.scrape_article_content(article, fetch_url), articles))

# Save the articles including full content to a CSV file
scraper.save_to_csv(articles)
