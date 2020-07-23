import pandas as pd
from bs4 import BeautifulSoup
import requests

url='http://openapi.seoul.go.kr:8088/<인증키>/xml/VwsmTrdarFlpopQq/1/1000/'
req=requests.get(url)
html=req.text
soup=BeautifulSoup(html, 'html.parser')
codenumber = soup.find_all('trdar_cd')

print(req)
#print(html)
print(codenumber)
