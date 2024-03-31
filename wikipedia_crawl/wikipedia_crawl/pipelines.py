# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import json
# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import logging
from scrapy.exceptions import DropItem

class WikipediaCrawlPipeline:
    def open_spider(self, spider):
        self.file = open('lable_3.json', 'w', encoding='utf-8')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        try:
            # Kiểm tra dữ liệu trước khi ghi
            if not item:
                raise DropItem("Missing item.")
            
            # Chuyển đổi dữ liệu thành JSON
            item_dict = ItemAdapter(item).asdict()
            json_line = json.dumps(item_dict, ensure_ascii=False) + "\n"
            
            # Ghi dữ liệu vào file JSON
            self.file.write(json_line)
        except Exception as e:
            # Xử lý ngoại lệ
            logging.error(f"Error processing item: {e}")
        finally:
            return item
