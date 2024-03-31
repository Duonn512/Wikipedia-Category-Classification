import requests
from lxml import html

def crawl(url):
    response = requests.get(url)
    tree = html.fromstring(response.content)
    
    # Get content
    content_parts = []
    paragraphs = tree.xpath('//*[@id="mw-content-text"]/div[1]/p[1]//text()')
    if paragraphs and paragraphs not in [[' \n'],['\n'],['\n\n']] :
        for node in paragraphs :
            if node.strip():
                content_parts.append(node.strip())
    else:
        paragraphs = tree.xpath('//*[@id="mw-content-text"]/div[1]/p[2]//text()')
        for node in paragraphs:
            if node.strip():
                content_parts.append(node.strip())

    content = ' '.join(content_parts)
    
    # Get title
    title = tree.xpath('//h1[@id="firstHeading"]//text()')
    if title:
        title = title[0]
    else:
        title = None

    return {'title': title, 'content': content}

def main():
    url = input("Wikipedia URL: ")
    result = crawl(url)
    print(result)
