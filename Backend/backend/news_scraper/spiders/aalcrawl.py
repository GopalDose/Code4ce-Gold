from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor


class CrawlingSpider(CrawlSpider):
    name = "aalcrawl"
    allowed_domains = ["aljazeera.com"]
    start_urls = ["https://www.aljazeera.com/"]

    rules = (
        Rule(LinkExtractor(allow=r"/news/\d{4}/"), callback="parse_item", follow=False),
    )

    def parse_item(self, response):
        yield {
            "title": response.css("main h1::text").get(),
            "desc": response.css("p.article__subhead em::text").get(),
            "image": response.css("figure img::attr(src)").get(),
            "content": response.css('div.wysiwyg p::text').getall(),  
            "url": response.url,
        }
