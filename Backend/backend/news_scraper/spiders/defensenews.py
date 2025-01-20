from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class CrawlingSpider(CrawlSpider):
    name = "defencecom"
    allowed_domains = ["defensenews.com"]
    start_urls = ["https://www.defensenews.com/"]

    rules = (
        Rule(LinkExtractor(allow=r"/global/"), callback="parse_item", follow=False),
    )

    def parse_item(self, response):
        yield {
            "title": response.css("h1.ClampedBox-sc-1pg0sor-0::text").get(),
            "desc": response.css("figcaption.a-caption::text").get(),
            "image": response.css("img.c-image::attr(src)").get(),
            "content": response.css('article.articleBody p::text').getall(),  
            "url": response.url,
        }
