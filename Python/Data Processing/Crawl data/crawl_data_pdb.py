import bs4
import pandas as pd
import requests
url = 'https://www.imdb.com/search/title/?count=100&groups=top_1000&sort=user_rating%27'

def get_page_content(url):
   page = requests.get(url,headers={"Accept-Language":"en-US"})
   return bs4.BeautifulSoup(page.text,"html.parser")
   
soup = get_page_content(url)
# print(soup.text)
movies = soup.findAll('h3', class_='lister-item-header')
# for movie in movies:
# 	titles = [movie.find('a').text]
# 	release = [movie.find('span',class_="lister-item-year text-muted unbold").text]
# 	rate = [movie.find('div', 'inline-block ratings-imdb-rating')]
# 	# titles.append(titles)
# print(movies)
titles = [movie.find('a').text for movie in movies]
release = [movie.find('span',class_="lister-item-year text-muted unbold").text for movie in movies]
# rate = [movie.find('div', 'inline-block ratings-imdb-rating') for movie in movies]
# print(titles)
# print(release)
# print(rate)
# print(soup.findAll('span',class_='certificate'))
certificate = [ce.text for ce in soup.findAll('span',class_='certificate')]
runtime = [rt.text for rt in soup.findAll('span',class_='runtime')]
# print(certificate)
# print(runtime)
# print(soup.findAll('div',class_='inline-block ratings-imdb-rating'))
rates = [rate['data-value'] for rate in soup.findAll('div',class_='inline-block ratings-imdb-rating')]
# print(rates)
# voting = [votes['data-value'] for votes in soup.findAll('span',{'name':'nv'})]
# votes = [vote['data-value'] for vote in soup.findAll('span',{'name':'nv'})]
# print(votes)
# votes = [vote['data-value'] for vote in voting]
# print(voting)
movies = soup.findAll('div', class_='lister-item-content')
# print(movies)
directors = [movie.find('p',class_='').find('a',class_='').text for movie in movies]
# print(director)
# director = soup.findAll('p')
# directors = [di.find('a').text for di in director]
# print(directors)
votes = [movie.findAll('span',{'name':'nv'})[0]['data-value'] for movie in movies]
earnings = [movie.findAll('span',{'name':'nv'})[-1]['data-value'] for movie in movies]
# print(len(earnings))
# print(len(votes))
# actors = [actor.text for actor in movie.find('p',class_='').findAll('a',class_='')[1:] for movie in movies]
# actors=[]
# for movie in movies:
# 	movie = [movie.find('p',class_='').findAll('a',class_='')[1:]]
actors = []
for movie in movies:
	# actors = [actor.text for actor in movie.find('p',class_='').findAll('a',class_='')[1:]]
	actors.append([actor.text for actor in movie.find('p',class_='').findAll('a',class_='')[1:]])
# print(len(actors))

df_dict = {'Title': titles, 'Release': release, 'Certificate': certificate,
           'Runtime': runtime, 'IMDB Rating': rates,
           'Votes': votes, 'Box Office Earnings': earnings, 'Director': directors,
           'Actors': actors}

print(df_dict)

df = pd.DataFrame.from_dict(df_dict, orient='index')

print(df)

df = df.transpose()

print(df)

df.to_csv('imdb.csv') # index=False: not get index, encoding='utf-8': if encodeError