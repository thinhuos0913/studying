import urllib.request
import re

url = urllib.request.urlopen('https://finance.yahoo.com/quote/AAPL?ltr=1')

s = str(url.read())
print(len(s))
print(s)

reg = """<fin-streamer>(,.+?)</fin-streamer>"""

# reg = """<span class="price up">(+.?)</span>==0$"""

pattern = re.compile(reg)

price = re.findall(pattern,s)

print(price)