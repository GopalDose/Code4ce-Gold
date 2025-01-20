from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class CrawlingSpider(CrawlSpider):
    name = "indian"
    allowed_domains = ["indianexpress.com"]
    start_urls = ["https://indianexpress.com/about/defence/"]

    rules = (
        Rule(LinkExtractor(allow=r"/"), callback="parse_item", follow=False),
    )

    def parse_item(self, response):
        yield {
            "title": response.css("h1.native_story_title::text").get(),
            "desc": response.css("h2.synopsis::text").get(),
            "image": response.css("span.custom-caption img::attr(src)").get(),
            "content": response.css('div.story_details p::text').getall(),  
            "url": response.url,
        }