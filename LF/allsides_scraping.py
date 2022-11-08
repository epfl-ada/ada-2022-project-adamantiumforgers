#%%
import os
import time
import pandas as pd

from bs4 import BeautifulSoup
from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

#%%
url = "https://www.allsides.com/media-bias/ratings"
url_basis="https://www.allsides.com"
page = urlopen(url)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")
#%%
#A="C:/Users/fisch/Desktop"
#B='chromedriver'
#chromrdriver =os.path.join(A,B)
#os.environ["webdriver.chrome.driver"] = chromrdriver
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
test=[]
href=[]
name=[]
orr=[]
com_agr_num=[]
com_disagr_num=[]

for mydiv in soup_mydivs:
    var=mydiv.get_text()
    var=re.sub("\(.*?\)","",var)
    var=var.strip()
    name.append(var)

for mydiv in soup_mydivs:
    var=mydiv.find('a').get('href')
    href.append(var)
    url_media=url_basis+var
    page = urlopen(url_media)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    soup_conf = soup.find("ul",{"class": "b-list"})
    sps=soup_conf.find("li",{"class":["Medium","High","Low or Initial","na"]})
    sp=sps.get("class")
    sp=str(sp).replace('[','').replace(']','').replace("'",'').replace(',','')
    test.append(sp)


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



# %%
df_media=pd.DataFrame
df_media=pd.read_csv('media_ft_man_mod copy.csv')
df_media=df_media.drop(columns=['Unnamed: 0','confidence','orrientation','comu_agree','comu_disagree'])
df_media['orrientation']=orr
df_media['confidence']=test
df_media['comu_agree']=com_agr_num
df_media['comu_disagree']=com_disagr_num
df_media=df_media.drop_duplicates(['name'])
df_media.to_csv('media_ft_man_mod_2.csv')



# %%
