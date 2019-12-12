from icrawler.builtin import GoogleImageCrawler
google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': 'action camera'})
google_crawler.session.verify = False
google_crawler.crawl(keyword='action camera', max_num=200,
                     min_size=(10,10), max_size=None)