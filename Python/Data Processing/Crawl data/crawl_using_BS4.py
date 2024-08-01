import bs4
import pandas as pd
import requests
url = 'https://www.imdb.com/search/title/?count=100&groups=top_1000&sort=user_rating%27'

def get_page_content(url):
   page = requests.get(url,headers={"Accept-Language":"en-US"})
   return bs4.BeautifulSoup(page.text,"html.parser")
   
soup = get_page_content(url)
# print(soup.prettify())
# print(soup.text)
# print(soup.status_code)
# results=soup.findAll('div',class_='inline-block ratings-imdb-rating')
# movies = soup.findAll('h3', class_='lister-item-header')
# for movie in movies:
# 	titles = movie.find('a').text
# 	release = movie.find('span',class_="lister-item-year text-muted unbold").text
# 	rate = movie.find('div', 'inline-block ratings-imdb-rating')
	# print(titles)
	# print(release)
	# print(rate)
# for ce in soup.findAll('span',class_='certificate'): 
# 	certificate = ce.text
	# print(certificate)
# for rt in soup.findAll('span', class_='runtime'):
# 	runtime = rt.text
	# print(runtime)
# for gr in soup.findAll('span',class_='genre'):
# 	genre = gr.text
	# print(genre)
# for rate in soup.findAll('div',class_='inline-block ratings-imdb-rating'):
# 	rates = rate['data-value']
	# print(rate)
# movies = soup.findAll('h3', class_='lister-item-header')
# titles = [movie.find('a').text for movie in movies]
# release = [rs.find('span',class_="lister-item-year text-muted unbold").text for rs in movies]
# # rate = movie.find('div', 'inline-block ratings-imdb-rating')
# certificate = [ce.text for ce in soup.findAll('span',class_='certificate')]
# runtime = [rt.text for rt in soup.findAll('span',class_='runtime')]
# genre = [gr.text for gr in soup.findAll('span',class_="genre")]
# rates = [rate['data-value'] for rate in soup.findAll('div',class_='inline-block ratings-imdb-rating')]
# Show data:
# df=pandas.DataFrame({'titles':[titles],'release':[release],'certificate':[certificate],'runtime':[runtime],'genre':[genre],'rates':[rates]})
# print(df)
# df=pandas.DataFrame({'titles':titles, 
#                   'release':release, 
#                   'certificate':certificate,
#                   'runtime':runtime,
#                   'genre': genre,
#                   'rates': rates})
# print(df)
# SAVE DATAFRAME INTO .CSV (.TXT) FILE:
# with open('scraped_text.txt', 'w') as file:
#     file.write(df)
# df.to_csv('scraped_data.csv') # index=False: not get index, encoding='utf-8': if encodeError
# DEFINE FUNCTION TO ALL:
# import bs4
# import pandas as pd
# import requests

def numeric_value(movie, tag, class_=None, order=None):
    if order:
        if len(movie.findAll(tag, class_)) > 1:
            to_extract = movie.findAll(tag, class_)[order]['data-value']
        else:
            to_extract = None
    else:
        to_extract = movie.find(tag, class_)['data-value']

    return to_extract

def text_value(movie, tag, class_=None):
    if movie.find(tag, class_):
        return movie.find(tag, class_).text
    else:
        return

def nested_text_value(movie, tag_1, class_1, tag_2, class_2, order=None):
    if not order:
        return movie.find(tag_1, class_1).find(tag_2, class_2).text
    else:
        return [val.text for val in movie.find(tag_1, class_1).findAll(tag_2, class_2)[order]]

def extract_attribute(soup, tag_1, class_1='', tag_2='', class_2='',
                      text_attribute=True, order=None, nested=False):
    movies = soup.findAll('div', class_='lister-item-content')
    data_list = []
    for movie in movies:
        if text_attribute:
            if nested:
                data_list.append(nested_text_value(movie, tag_1, class_1, tag_2, class_2, order))
            else:
                data_list.append(text_value(movie, tag_1, class_1))
        else:
            data_list.append(numeric_value(movie, tag_1, class_1, order))

    return data_list

titles = extract_attribute(soup, 'a')
release = extract_attribute(soup, 'span', 'lister-item-year text-muted unbold')
audience_rating = extract_attribute(soup, 'span', 'certificate')
runtime = extract_attribute(soup, 'span', 'runtime')
genre = extract_attribute(soup, 'span', 'genre')
imdb_rating = extract_attribute(soup, 'div', 'inline-block ratings-imdb-rating', False)
votes = extract_attribute(soup, 'span' , {'name' : 'nv'}, False, 0)
earnings = extract_attribute(soup, 'span' , {'name' : 'nv'}, False, 1)
directors = extract_attribute(soup, 'p', '', 'a', '', True, 0, True)
actors = extract_attribute(soup, 'p', '', 'a', '', True, slice(1, 5, None), True)


df_dict = {'Title': titles, 'Relase': release, 'Audience Rating': audience_rating,
           'Runtime': runtime, 'Genre': genre, 'IMDB Rating': imdb_rating,
           'Votes': votes, 'Box Office Earnings': earnings, 'Director': directors,
           'Actors': actors}
df = pd.DataFrame(df_dict)
print(df)
df.to_csv('scraped_data1.csv') # index=False: not get index, encoding='utf-8': if encodeError
