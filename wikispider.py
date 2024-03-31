import scrapy
import csv

class WikispiderSpider(scrapy.Spider):
    name = "wikispider"
    allowed_domains = ["vi.wikipedia.org"]
    start_urls = ['https://vi.wikipedia.org']

    def start_requests(self):
        csv_file_path = r'C:\Users\Admin\test_crawl\wikipedia_crawl\wikipedia_crawl\spiders\Label_3.csv'
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                url = row[0].strip()
                yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        title = response.css('#firstHeading ::text').get()
        content_parts = []
        r1 = response.xpath('//*[@id="mw-content-text"]/div[1]/p[1]//text() ').getall()
        if r1 != [' \n'] and r1 != ['\n'] and r1 != ['\n\n'] :
            for node in r1 :
                if node.strip():
                    content_parts.append(node.strip())

            content = ' '.join(content_parts)    
        else :
            for node in response.xpath('//*[@id="mw-content-text"]/div[1]/p[2]//text() ').getall() :
                if node.strip():
                    content_parts.append(node.strip())
            content = ' '.join(content_parts)

        yield {
            'title': title,
            'content' : content,
        }