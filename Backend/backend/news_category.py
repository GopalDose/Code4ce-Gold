import json
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import OpenAI
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime
from pathlib import Path
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import hashlib
from dataclasses import dataclass, asdict
import numpy as np

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('news_processor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class Article:
    """Data class for article structure"""
    title: str
    content: List[str]
    desc: Optional[str] = None
    image: Optional[str] = None
    url: Optional[str] = None
    timestamp: Optional[str] = None

@dataclass
class ProcessedArticle:
    """Data class for processed article with category"""
    article: Article
    category: str
    confidence_score: float
    processing_time: float
    processed_at: str

class CategoryValidator:
    """Validates and normalizes category assignments"""
    
    VALID_CATEGORIES = {
        'politics', 'technology', 'business', 'sports', 'entertainment',
        'health', 'science', 'education', 'environment', 'world'
    }

    @staticmethod
    def normalize_category(category: str) -> str:
        """Normalizes category string and validates against known categories"""
        normalized = category.lower().strip()
        
        # Simple fuzzy matching
        for valid_category in CategoryValidator.VALID_CATEGORIES:
            if normalized in valid_category or valid_category in normalized:
                return valid_category
                
        return 'other'

class NewsCategoryProcessor:
    def __init__(self, temperature: float = 0.2, max_workers: int = 4):
        """
        Initialize the NewsCategoryProcessor with enhanced configuration.
        
        Args:
            temperature: Float value for LLM temperature
            max_workers: Maximum number of concurrent workers
        """
        self._initialize_environment()
        self.llm = self._setup_llm(temperature)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def _initialize_environment(self) -> None:
        """Initialize environment variables and create necessary directories"""
        load_dotenv()
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        # Create directories for output and logs
        os.makedirs('output', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def _setup_llm(self, temperature: float) -> OpenAI:
        """Setup LLM with error handling"""
        try:
            return OpenAI(temperature=temperature)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: logger.info(f"Retrying after error: {retry_state.outcome.exception()}")
    )
    async def categorize_article(self, article: Article) -> Tuple[str, float]:
        """
        Categorize an article using the LLM with enhanced prompt engineering.
        
        Returns:
            Tuple containing (category, confidence_score)
        """
        start_time = datetime.now()
        
        prompt = self._generate_prompt(article)
        
        try:
            response = self.llm.invoke(prompt)
            category = CategoryValidator.normalize_category(response)
            
            # Simple confidence score based on prompt length and response time
            confidence_score = self._calculate_confidence_score(
                response_time=(datetime.now() - start_time).total_seconds(),
                prompt_length=len(prompt)
            )
            
            return category, confidence_score
            
        except Exception as e:
            logger.error(f"Error categorizing article: {article.title}. Error: {str(e)}")
            raise

    def _generate_prompt(self, article: Article) -> str:
        """Generate a detailed prompt for the LLM"""
        return (
            "As an expert news analyst, categorize the following article into exactly one of these "
            f"categories: {', '.join(CategoryValidator.VALID_CATEGORIES)}.\n\n"
            f"Title: {article.title}\n"
            f"Description: {article.desc or 'No description available.'}\n"
            f"Content Summary: {' '.join(article.content[:3])}...\n\n"
            "Provide only the category name, nothing else."
        )

    def _calculate_confidence_score(self, response_time: float, prompt_length: int) -> float:
        """Calculate a confidence score based on various metrics"""
        # Simplified score calculation
        base_score = 0.8
        time_penalty = min(0.2, response_time / 10)
        length_factor = min(0.2, prompt_length / 1000)
        
        return round(base_score - time_penalty + length_factor, 2)

    async def process_and_save_articles(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 10
    ) -> None:
        """Process articles in batches asynchronously"""
        try:
            articles = await self._load_articles(input_file)
            results = []
            
            # Process articles in batches
            for i in range(0, len(articles), batch_size):
                batch = articles[i:i + batch_size]
                batch_results = await asyncio.gather(
                    *[self._process_single_article(Article(**article)) for article in batch]
                )
                results.extend(batch_results)
                
                # Intermediate save
                await self._save_results(results, output_file)
                logger.info(f"Processed and saved batch {i//batch_size + 1}")
                
            # Final save with statistics
            await self._save_results(results, output_file, include_stats=True)
            
        except Exception as e:
            logger.error(f"Failed to process articles: {str(e)}")
            raise

    async def _load_articles(self, input_file: str) -> List[Dict[str, Any]]:
        """Load articles from file with validation"""
        try:
            async with aiofiles.open(input_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                articles = json.loads(content)
                
            if not isinstance(articles, list):
                raise ValueError("Input file must contain a list of articles")
                
            return articles
            
        except Exception as e:
            logger.error(f"Error loading articles: {str(e)}")
            raise

    async def _process_single_article(self, article: Article) -> ProcessedArticle:
        """Process a single article with timing and error handling"""
        start_time = datetime.now()
        
        try:
            category, confidence_score = await self.categorize_article(article)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessedArticle(
                article=article,
                category=category,
                confidence_score=confidence_score,
                processing_time=processing_time,
                processed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing article {article.title}: {str(e)}")
            return ProcessedArticle(
                article=article,
                category="error",
                confidence_score=0.0,
                processing_time=0.0,
                processed_at=datetime.now().isoformat()
            )

    async def _save_results(
        self,
        results: List[ProcessedArticle],
        output_file: str,
        include_stats: bool = False
    ) -> None:
        """Save results with optional statistics"""
        output_data = {
            "processed_articles": [asdict(result) for result in results],
            "metadata": {
                "total_articles": len(results),
                "processed_at": datetime.now().isoformat(),
                "success_rate": self._calculate_success_rate(results)
            }
        }
        
        if include_stats:
            output_data["statistics"] = self._generate_statistics(results)
        
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(output_data, indent=4))

    def _calculate_success_rate(self, results: List[ProcessedArticle]) -> float:
        """Calculate the success rate of processing"""
        successful = sum(1 for result in results if result.category != "error")
        return round(successful / len(results) * 100, 2)

    def _generate_statistics(self, results: List[ProcessedArticle]) -> Dict[str, Any]:
        """Generate detailed statistics about the processing results"""
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        confidence_scores = [r.confidence_score for r in results if r.confidence_score > 0]
        
        return {
            "category_distribution": self._get_category_distribution(results),
            "processing_time": {
                "mean": np.mean(processing_times),
                "median": np.median(processing_times),
                "std": np.std(processing_times)
            },
            "confidence_scores": {
                "mean": np.mean(confidence_scores),
                "median": np.median(confidence_scores),
                "std": np.std(confidence_scores)
            }
        }

    def _get_category_distribution(self, results: List[ProcessedArticle]) -> Dict[str, int]:
        """Calculate the distribution of categories"""
        distribution = {}
        for result in results:
            distribution[result.category] = distribution.get(result.category, 0) + 1
        return distribution

# Main execution
if __name__ == "__main__":
    INPUT_FILE = Path("data.json")
    OUTPUT_FILE = Path("categorized_articles.json")
    
    async def main():
        processor = NewsCategoryProcessor()
        await processor.process_and_save_articles(
            str(INPUT_FILE),
            str(OUTPUT_FILE),
            batch_size=10
        )
    
    asyncio.run(main())