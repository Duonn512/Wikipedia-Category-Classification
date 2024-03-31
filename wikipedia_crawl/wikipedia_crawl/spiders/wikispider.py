import scrapy

class WikispiderSpider(scrapy.Spider):
    name = "wikispider"
    allowed_domains = ["vi.wikipedia.org"]
    start_urls = ['https://vi.wikipedia.org/wiki/Đặc_biệt:Ngẫu_nhiên']
    collected_count = 0
    number_wiki_collected = 1000

    def parse(self, response):
        if self.collected_count < self.number_wiki_collected:
            title = response.css('#firstHeading ::text').get()

            # Take first content
            content_parts = []
            r1 = response.xpath('//*[@id="mw-content-text"]/div[1]/p[1]//text()')
            if r1 not in [[' \n'], ['\n'], ['\n\n']]:
                for node in r1:
                    if node.strip():
                        content_parts.append(node.strip())
            else:
                r2 = response.xpath('//*[@id="mw-content-text"]/div[1]/p[2]//text()')
                for node in r2:
                    if node.strip():
                        content_parts.append(node.strip())

            content = ' '.join(content_parts)
            self.collected_count += 1

            # Take next content
            next_content = response.css('#n-randompage a::attr(href)').get()
            if next_content:
                next_content_url = 'https://vi.wikipedia.org' + next_content
                yield response.follow(next_content_url, callback=self.parse, dont_filter=True)
