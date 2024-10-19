from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import csv

class AlJazeeraEnglishScraper:
    base_url = "https://www.bbc.com"

    def get_latest_articles(self, fetch_url): 
        content = fetch_url(self.base_url + '/news/')
        if content is None:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        articles = []

        for item in soup.find_all('article'):
            title = item.find('h3').get_text() if item.find('h3') else 'No title'
            link = item.find('a')['href'] if item.find('a') else None
            summary = item.find('p').get_text() if item.find('p') else 'No summary'
            category = item.find('div', class_='post-category').get_text() if item.find('div', class_='post-category') else 'No category'
            image = item.find('img')['src'] if item.find('img') else 'No image'

            if link:
                full_article = self.scrape_article_content(fetch_url, f"{self.base_url}{link}")
                articles.append({
                    'title': title,
                    'link': f"{self.base_url}{link}",
                    'summary': summary,
                    'category': category,
                    'image': self.base_url + image,
                    'published_date': full_article.get('published_date', 'No date'),
                    'full_content': full_article.get('content', 'No content')
                })

        return articles

    def scrape_article_content(self, fetch_url, article_url):
        """Visit the article URL and scrape the full content."""
        content = fetch_url(article_url)
        if content is None:
            return {'content': 'No content', 'published_date': 'No date'}
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Scraping the full content from the article page
        article_content = soup.find('div', class_='wysiwyg').get_text(strip=True) if soup.find('div', class_='wysiwyg') else 'No content'

        # Scraping the published date
        published_date = soup.find('time')['datetime'] if soup.find('time') else 'No date'

        return {'content': article_content, 'published_date': published_date}

def fetch_url(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run Chrome in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        driver.get(url)
        time.sleep(5)  # Give the page time to load
        html = driver.page_source
        driver.quit()
        return html
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def save_to_csv(articles, filename='latest_articles.csv'):
    if not articles:
        print("No articles to save.")
        return
    
    keys = articles[0].keys()  # Get the column names from the article keys
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(articles)

# Main execution
scraper = AlJazeeraEnglishScraper()
html_content = fetch_url(scraper.base_url + '/news/')
latest_articles = scraper.get_latest_articles(fetch_url)

# Print the results and save them to a CSV file
if latest_articles:
    for article in latest_articles:
        print(article)

    # Save articles to CSV
    save_to_csv(latest_articles, 'latest_aljazeera_articles.csv')
    print(f"Saved {len(latest_articles)} articles to 'latest_aljazeera_articles.csv'")
else:
    print("No articles found.")
