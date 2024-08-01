# import newspaper
# from newspaper import news_pool
# Multi-threading article downloads
# slate_paper = newspaper.build('http://slate.com')
# tc_paper = newspaper.build('http://techcrunch.com')
# espn_paper = newspaper.build('http://espn.com')
# papers = [slate_paper, tc_paper, espn_paper]
# news_pool.set(papers, threads_per_source=2) # (3*2) = 6 threads total
# news_pool.join() # At this point, you can safely assume that download() has been called on every single article for all 3 sources.
# print(espn_paper.articles[10].html)
# Keeping Html of main body article
# from newspaper import Article
# a = Article('http://www.cnn.com/2014/01/12/world/asia/north-korea-charles-smith/index.html', keep_article_html=True)
# a.download()
# a.parse()
# print(a.article_html)
# print(a.clean_dom)
# print(a.clean_top_node)
# from newspaper import Source
# cnn_paper = Source('http://cnn.com')
# print(cnn_paper.size()) # no articles, we have not built the source
# cnn_paper.build()
# print(cnn_paper.size())
# cnn_paper.download()
# cnn_paper.parse()
# cnn_paper.set_categories()
# cnn_paper.download_categories()
# cnn_paper.parse_categories()
# cnn_paper.set_feeds()
# cnn_paper.download_feeds()
# cnn_paper.generate_articles()
# print(cnn_paper.size())
# from newspaper import Article, Source, Config
# Named parameter passing examples:
# cnn = newspaper.build('http://cnn.com', language='en', memoize_articles=False)
# article = Article(url='http://cnn.com/french/...', language='fr', fetch_images=False)
# cnn = Source(url='http://latino.cnn.com/...', language='es', request_timeout=10, number_threads=20)
# Here are some examples of how Config objects are passed:
# config = Config()
# config.memoize_articles = False
# cbs_paper = newspaper.build('http://cbs.com', config)
# article_1 = Article(url='http://espn/2013/09/...', config)
# cbs_paper = Source('http://cbs.com', config)
# import requests
# response = requests.get('https://api.github.com')
# print(response.content)
# print(response.text)
# print(response.json())
# print(response.headers)
# print(response.headers['Content-type'])
