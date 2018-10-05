# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import time
import urllib.request
import requests

count = 0

session = requests.Session()
while (count<=5000):
	count += 1    
	response = session.get('http://www.taifex.com.tw/cht/captcha', cookies={'from-my': 'browser'})
	with open('ig/'+str(count)+'.jpg', 'wb') as file:
		file.write(response.content)
		file.flush()
