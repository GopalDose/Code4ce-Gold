from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class CrawlingSpider(CrawlSpider):
    name = "defence"
    allowed_domains = ["defence.in"]
    start_urls = ["https://defence.in/"]

    rules = (
        Rule(LinkExtractor(allow=r"/threads/"), callback="parse_item", follow=False),
    )

    def parse_item(self, response):
        yield {
            "title": response.css("h1.p-title-value::text").get(),
            "desc": response.css("h1.p-title-value::text").get(),
            "image": response.css("div.bbImageWrapper img::attr(src)").get(),
            "content": response.css('div.bbWrapper::text').getall(),  
            "url": response.url,
        }
