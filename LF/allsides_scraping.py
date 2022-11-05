#%%


import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen

import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

url = "https://www.allsides.com/media-bias/ratings"
page = urlopen(url)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")
#%%
A="C:/Users/fisch/Desktop"
B='chromedriver'
chromrdriver =os.path.join(A,B)
os.environ["webdriver.chrome.driver"] = chromrdriver
driver = webdriver.Chrome(executable_path=r'C:/Users/fisch/Desktop/chromedriver/chromedriver.exe')
#driver.get("https://www.allsides.com/media-bias/ratings?field_featured_bias_rating_value=All&field_news_source_type_tid[1]=1&field_news_source_type_tid[2]=2&field_news_source_type_tid[3]=3&field_news_source_type_tid[4]=4")
driver.get(url)

ScrollNumber = 2
for i in range(1,ScrollNumber):
    driver.execute_script("window.scrollTo(1,1000000)")
    time.sleep(5)

    
file = open('DS_ft.html', 'w')
file.write(driver.page_source)
file.close()

driver.close()

test= open('DS_ft.html', 'r')
soup = BeautifulSoup(test, "html.parser")
test.close

#%%
soup_mydivs = soup.find_all("td", {"class": "views-field views-field-title source-title"})
soup_pol_or = soup.find_all("td", {"class": "views-field views-field-field-bias-image"})
soup_com_agr =soup.find_all("span",{"class": "agree"})
soup_com_disagr =soup.find_all("span",{"class": "disagree"})
#%%
import re

name=[]
orr=[]
com_agr_num=[]
com_disagr_num=[]
for mydiv in soup_mydivs:
    var=mydiv.get_text()
    var=re.sub("\(.*?\)","",var)
    var=var.strip()
    name.append(var)
    

for pol_o in soup_pol_or:  
    orr.append(pol_o.find('img').get('title')[28:])

for com_agr in soup_com_agr:
    truc2=com_agr.get_text()
    com_agr_num.append(truc2)
del com_agr_num[1::2]

for com_disagr in soup_com_disagr:
    truc1=com_disagr.get_text()
    com_disagr_num.append(truc1)
del com_disagr_num[1::2]

df_media=pd.DataFrame({'name':name,'orr':orr,'agr':com_agr_num,'disagr':com_disagr_num})
df_media.to_csv('media_ft.csv')


