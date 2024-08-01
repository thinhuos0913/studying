from newspaper import Article

# url='https://vnexpress.net/tp-hcm-lap-to-cong-tac-giam-sat-chat-cac-khu-phong-toa-4330014.html'
# article=Article(url)
# article.download()
# article.parse()
# # Crawl data:
# print(article.title)
# print(article.authors)
# print(article.publish_date)
# print(article.text[:150])
# print(article.top_image)
# print(article.movies)
# print(article.nlp())
# print(article.keywords)
# print(article.summary)

import newspaper
import requests

vn_paper=newspaper.build('https://www.24h.com.vn/')
for article in vn_paper.articles:
	print('Link:\n', article.url)

for category in vn_paper.category_urls():
	print('Categories:\n', category)

# paper_article = vn_paper.articles[0]
# paper_article.download()
# paper_article.parse()
# paper_article.nlp()
# print('paper_article:\n', paper_article)

# from newspaper import fulltext
# html=requests.get(...).text
# text=fulltext(html)
# print(newspaper.languages())