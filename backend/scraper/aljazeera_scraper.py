from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import requests
import json

from models.article_model import save_article_to_db

class AlJazeeraEnglishScraper:
    base_url = "https://www.aljazeera.com"

    def get_latest_articles(self):
        content = self.fetch_url(self.base_url + '/news')
        if content is None:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        articles = []

        for item in soup.find_all('article'):
            title = item.find('h3').get_text() if item.find('h3') else 'No title'
            link = item.find('a')['href'] if item.find('a') else None
            summary = item.find('p').get_text() if item.find('p') else 'No summary'
            image = item.find('img')['src'] if item.find('img') else 'No image'
            published_date = item.find('span', aria_hidden='true').get_text() if item.find('span', aria_hidden='true') else 'No date'

            if link:
                articles.append({
                    'title': title,
                    'link': f"{self.base_url}{link}",
                    'summary': summary,
                    'image': self.base_url + image,
                    'published_date': published_date,
                })

        return articles

    def scrape_article_content(self, article):
        """Visit the article URL and scrape the full content and accurate published date."""
        content = self.fetch_url(article['link'])
        if content is None:
            article['full_content'] = 'No content'
            return article

        soup = BeautifulSoup(content, 'html.parser')
        article_content = soup.find('div', class_='wysiwyg').get_text(strip=True) if soup.find('div', class_='wysiwyg') else 'No content'
        article['full_content'] = article_content

        published_date = soup.find('time')['datetime'] if soup.find('time') else article['published_date']
        article['published_date'] = published_date

        # Summarize the full content if it's too long
        if len(article_content) > 250:
            print(f"Content is too long, summarizing: {len(article_content)} characters")
            article['full_content'] = self.summarize_large_content(article_content)
        else:
            article['full_content'] = article_content[:250] + "..."

        return article

    def fetch_url(self, url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def summarize_large_content(self, content, chunk_size=1024):
        """Split large content into smaller chunks and summarize each chunk."""
        # Split content into chunks of `chunk_size` characters
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        summarized_chunks = []

        # Summarize each chunk
        for chunk in chunks:
            summary = self.summarize_content(chunk)
            summarized_chunks.append(summary)

        # Combine all summarized chunks into a single summary
        return ' '.join(summarized_chunks)

    def summarize_content(self, content):
        """Use Hugging Face API to summarize the content."""
        api_url = 'https://api-inference.huggingface.co/models/facebook/bart-large-cnn'
        headers = {
            'Authorization': f'Bearer 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',
            'Content-Type': 'application/json'
        }
        payload = {
            'inputs': content
        }

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                # Inspect the response to ensure it's correct
                response_json = response.json()
                print(f"Summarization response: {response_json}")
                
                if 'summary_text' in response_json[0]:
                    summary = response_json[0]['summary_text']
                    return summary
                else:
                    print(f"Unexpected API response format: {response_json}")
                    return content[:250] + "..."  # fallback to a truncated version
            else:
                print(f"Failed to summarize content. Status code: {response.status_code}")
                return content[:250] + "..."  # fallback to a truncated version
        except Exception as e:
            print(f"Error summarizing content: {e}")
            return content[:250] + "..."  # fallback in case of failure

def run_scraper(db):
    """Run the scraper and save data to the provided MongoDB `db` instance."""
    scraper = AlJazeeraEnglishScraper()
    latest_articles = scraper.get_latest_articles()

    if latest_articles:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(scraper.scrape_article_content, article) for article in latest_articles]
            for future in as_completed(futures):
                article = future.result()
                save_article_to_db(article, db)

        print(f"Scraped and saved {len(latest_articles)} articles.")
    else:
        print("No new articles found.")
