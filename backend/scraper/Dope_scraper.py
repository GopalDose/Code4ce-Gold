from typing import Dict, List
from dataclasses import dataclass
from pymongo import MongoClient
import os
from abc import ABC, abstractmethod
from robust_scraper import RobustScraper, ScrapingConfig, Article

# Abstract base class for all news scrapers
class NewsScraper(ABC):
    def __init__(self, base_url: str, config: ScrapingConfig):
        self.base_url = base_url
        self.scraper = RobustScraper(config)
        self.selectors = self.get_selectors()

    @abstractmethod
    def get_selectors(self) -> Dict:
        pass

# Define all scrapers with specific configurations
class HinduScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['thehindu.com'], rate_limit_delay=(2, 4))
        super().__init__('https://www.thehindu.com', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'div.story-card',
            'title': 'h3.title a',
            'summary': 'p.intro',
            'content': 'div.article-content',
            'category': 'a.section-name',
            'date_selectors': ['meta[property="article:published_time"]'],
            'date_formats': ['%Y-%m-%dT%H:%M:%S%z'],
            'tags': 'div.topic-tags a',
            'metadata': {'author': 'a.auth-nm', 'location': 'span.dateline'},
            'fetch_full_content': True
        }

class IndianExpressScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['indianexpress.com'], rate_limit_delay=(2, 4))
        super().__init__('https://indianexpress.com', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'div.article-list',
            'title': 'h2.title a',
            'summary': 'p.preview',
            'content': 'div.full-details',
            'category': 'div.category a',
            'date_selectors': ['span.date'],
            'date_formats': ['%B %d, %Y %I:%M %p'],
            'tags': 'div.tags a',
            'metadata': {'author': 'div.writer a', 'location': 'div.location'},
            'fetch_full_content': True
        }

class TimesOfIndiaScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['timesofindia.indiatimes.com'], rate_limit_delay=(2, 4))
        super().__init__('https://timesofindia.indiatimes.com', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'div.list-article',
            'title': 'span.title a',
            'summary': 'p.synopsis',
            'content': 'div.article-content',
            'category': 'span.category a',
            'date_selectors': ['span.date-time'],
            'date_formats': ['%b %d, %Y, %I:%M %p'],
            'tags': 'div.keywords a',
            'metadata': {'author': 'span.author', 'location': 'span.location'},
            'fetch_full_content': True
        }

class BBCScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['bbc.com', 'bbc.co.uk'], rate_limit_delay=(3, 5))
        super().__init__('https://www.bbc.com/news', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'div.gs-c-promo',
            'title': 'h3.gs-c-promo-heading',
            'summary': 'p.gs-c-promo-summary',
            'content': 'article.content',
            'category': 'a.gs-c-section-link',
            'date_selectors': ['time'],
            'date_formats': ['%d %B %Y'],
            'tags': 'ul.tags-list li',
            'metadata': {'author': 'div.byline', 'location': 'span.location'},
            'fetch_full_content': True
        }

class ReutersScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['reuters.com'], rate_limit_delay=(2, 4))
        super().__init__('https://www.reuters.com', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'div.story-content',
            'title': 'h3.story-title',
            'summary': 'p.story-summary',
            'content': 'div.ArticleBody__content',
            'category': 'a.section-link',
            'date_selectors': ['meta[property="article:published_time"]'],
            'date_formats': ['%Y-%m-%dT%H:%M:%S%z'],
            'tags': 'ul.tag-list a',
            'metadata': {'author': 'span.byline', 'location': 'span.location'},
            'fetch_full_content': True
        }

class AlJazeeraScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['aljazeera.com'], rate_limit_delay=(2, 4))
        super().__init__('https://www.aljazeera.com', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'article',
            'title': 'h1.article-heading',
            'summary': 'div.article-preamble p',
            'content': 'div.article-body',
            'category': 'div.article-section a',
            'date_selectors': ['time[datetime]'],
            'date_formats': ['%Y-%m-%dT%H:%M:%S%z'],
            'tags': 'ul.article-tags a',
            'metadata': {'author': 'span.author-name', 'location': 'span.article-location'},
            'fetch_full_content': True
        }

class ElPaisScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['elpais.com'], rate_limit_delay=(2, 4))
        super().__init__('https://elpais.com', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'div.articulo',
            'title': 'h2.articulo-titulo a',
            'summary': 'p.articulo-entradilla',
            'content': 'div.articulo-cuerpo',
            'category': 'a.articulo-seccion',
            'date_selectors': ['time[datetime]'],
            'date_formats': ['%Y-%m-%dT%H:%M:%S%z'],
            'tags': 'ul.tags-list a',
            'metadata': {'author': 'span.author-name', 'location': 'span.location'},
            'fetch_full_content': True
        }

class LeMondeScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['lemonde.fr'], rate_limit_delay=(2, 4))
        super().__init__('https://www.lemonde.fr', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'article',
            'title': 'h1.article-title',
            'summary': 'p.article-summary',
            'content': 'div.article-body',
            'category': 'div.section-link a',
            'date_selectors': ['meta[property="article:published_time"]'],
            'date_formats': ['%Y-%m-%dT%H:%M:%S%z'],
            'tags': 'ul.tags-list a',
            'metadata': {'author': 'span.author-name', 'location': 'span.location'},
            'fetch_full_content': True
        }

class DerSpiegelScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['spiegel.de'], rate_limit_delay=(2, 4))
        super().__init__('https://www.spiegel.de', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'article',
            'title': 'h2.article-title',
            'summary': 'p.article-summary',
            'content': 'div.article-body',
            'category': 'a.section-link',
            'date_selectors': ['meta[property="article:published_time"]'],
            'date_formats': ['%Y-%m-%dT%H:%M:%S%z'],
            'tags': 'ul.tags-list a',
            'metadata': {'author': 'span.author-name', 'location': 'span.location'},
            'fetch_full_content': True
        }

class RTScraper(NewsScraper):
    def __init__(self):
        config = ScrapingConfig(allowed_domains=['rt.com'], rate_limit_delay=(2, 4))
        super().__init__('https://www.rt.com', config)

    def get_selectors(self) -> Dict:
        return {
            'article_container': 'div.card',
            'title': 'h3.card-title a',
            'summary': 'p.card-summary',
            'content': 'div.article-body',
            'category': 'a.section-link',
            'date_selectors': ['time[datetime]'],
            'date_formats': ['%Y-%m-%dT%H:%M:%S%z'],
            'tags': 'ul.tags-list a',
            'metadata': {'author': 'span.author-name', 'location': 'span.location'},
            'fetch_full_content': True
        }

# Add other scrapers for JapanTimes, ChinaDaily, DeutscheWelle, BusinessStandardScraper, TheWireScraper, etc...

# Aggregator class for managing all scrapers
class NewsAggregator:
    def __init__(self):
        self.scrapers = [
            HinduScraper(),
            IndianExpressScraper(),
            TimesOfIndiaScraper(),
            BBCScraper(),
            ReutersScraper(),
            AlJazeeraScraper(),
            ElPaisScraper(),
            LeMondeScraper(),
            DerSpiegelScraper(),
            RTScraper(),
            # Add remaining scrapers here...
        ]

        self.mongo_user = os.getenv("MONGO_USER", "your_default_user")
        self.mongo_password = os.getenv("MONGO_PASSWORD", "your_default_password")
        self.mongo_cluster = "cluster0.vgpwq.mongodb.net"
        self.database_name = "news_aggregator"
        self.collection_name = "articles"

    def get_database_connection(self):
        mongo_uri = f"mongodb+srv://{self.mongo_user}:{self.mongo_password}@{self.mongo_cluster}/{self.database_name}?retryWrites=true&w=majority"
        client = MongoClient(mongo_uri)
        return client[self.database_name]

    def insert_articles(self, db, articles: List[Article]):
        if not articles:
            print("No articles to insert.")
            return

        try:
            collection = db[self.collection_name]
            article_dicts = [{
                'title': article.title,
                'url': article.url,
                'summary': article.summary,
                'content': article.content,
                'category': article.category,
                'published_date': article.published_date,
                'source': article.metadata.get('source'),
                'author': article.author,
                'tags': article.tags,
                'metadata': article.metadata
            } for article in articles]

            collection.insert_many(article_dicts)
            print(f"Inserted {len(articles)} articles into MongoDB.")
        except Exception as e:
            print(f"Error inserting articles into MongoDB: {e}")

    def run(self):
        all_articles = []
        db = self.get_database_connection()

        for scraper in self.scrapers:
            try:
                print(f"Scraping from: {scraper.base_url}")
                articles = scraper.scraper.scrape_website(scraper.base_url, scraper.selectors)

                for article in articles:
                    if article.metadata is None:
                        article.metadata = {}
                    article.metadata['source'] = scraper.base_url

                all_articles.extend(articles)
                print(f"Scraped {len(articles)} articles from {scraper.base_url}")

            except Exception as e:
                print(f"Error scraping {scraper.base_url}: {e}")

        self.insert_articles(db, all_articles)
        print(f"Total articles scraped and saved: {len(all_articles)}")

if __name__ == "__main__":
    aggregator = NewsAggregator()
    aggregator.run()
