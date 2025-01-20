import os
import json
import logging
import torch
import asyncio
import aiofiles
from typing import List, Dict, Any
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datetime import datetime
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FastArticleProcessor:
    def __init__(
        self,
        model_name: str = 'sshleifer/distilbart-cnn-6-6',  # Using a lighter model
        batch_size: int = 4,
        max_input_length: int = 512,
        target_summary_length: int = 150,
        num_workers: int = 2
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.target_summary_length = target_summary_length
        self.num_workers = num_workers
        self._setup_model()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def _setup_model(self) -> None:
        """Initialize model with performance optimizations"""
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True
        )
        
        # Move model to device
        if self.device.type == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
            
        self.model.eval()
        logger.info("Model loaded successfully")

    def _prepare_batch(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Prepare a batch of articles for processing"""
        contents = []
        for article in articles:
            content = article['article']['content']
            if isinstance(content, list):
                content = ' '.join(str(item) for item in content)
            category = article.get('category', 'general')
            prompt = f"Summarize this {category} article:\n{content}"
            contents.append(prompt)
        return contents

    @torch.no_grad()
    def _generate_summaries_batch(self, texts: List[str]) -> List[str]:
        """Generate summaries for a batch of texts"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Generate
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=self.target_summary_length,
            min_length=int(self.target_summary_length * 0.6),
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        # Decode
        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [' '.join(summary.split()) for summary in summaries] 

    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a single text."""
        try:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            compound_score = sentiment_scores['compound']

            if compound_score >= 0.05:
                sentiment = "positive"
            elif compound_score <= -0.05: 
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "overall_score": compound_score,
                "overall_label": sentiment,
                "details": sentiment_scores,
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {"overall_score": 0, "overall_label": "neutral", "details": {}}

    async def process_articles(self, input_file: str, output_file: str) -> None:
        """Process articles in batches"""
        try:
            async with aiofiles.open(input_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)

            articles = data['processed_articles'] if isinstance(data, dict) else data
            total_articles = len(articles)
            logger.info(f"Processing {total_articles} articles in batches of {self.batch_size}")

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for i in range(0, total_articles, self.batch_size):
                    batch = articles[i:i + self.batch_size]
                    batch_texts = self._prepare_batch(batch)
                    
                    loop = asyncio.get_event_loop()
                    try:
                        summaries = await loop.run_in_executor(
                            executor,
                            partial(self._generate_summaries_batch, batch_texts)
                        )
                        
                        for article, summary in zip(batch, summaries):
                            article['article']['summary'] = summary
                            sentiment = self._analyze_sentiment(summary)
                            article['article']['sentiment'] = sentiment
                        
                        logger.info(f"Processed batch {i//self.batch_size + 1}/{(total_articles + self.batch_size - 1)//self.batch_size}")
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        continue

            output_data = {
                "processed_articles": articles,
                "metadata": {
                    "total_articles": total_articles,
                    "processed_at": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "batch_size": self.batch_size
                }
            }

            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(output_data, indent=4, ensure_ascii=False))

            logger.info(f"Successfully processed {total_articles} articles")
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise

async def main():
    INPUT_FILE = "categorized_articles.json"
    OUTPUT_FILE = "summarized_articles.json"
    
    try:
        processor = FastArticleProcessor(
            model_name='sshleifer/distilbart-cnn-6-6',  # Lighter model
            batch_size=4,                               # Smaller batch size
            max_input_length=512,
            target_summary_length=150,
            num_workers=2                               # Fewer workers
        )
        await processor.process_articles(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
