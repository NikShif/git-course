from datetime import datetime
import requests
from bs4 import BeautifulSoup as bs
import csv
import json
import numpy as np 
import os
import re

def view_count_to_num(view_count):
    if view_count[-1] == 'K':
        k = re.sub(r'[.]?(\d*)(?:\w)?', r'\1', view_count)
        return(int(k+'00'))
    else:
        k = re.sub(r'[.]?(\d*)(?:\w)?', r'\1', view_count)
        return(int(k+'000000'))
#НУжна ли функция времени? или лучше перевксти в секунды
def duration_to_time(time):
    return datetime.strptime(time, '%H:%M')
    
hdr = {
       'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
       'accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
       'x-frame-options' : 'SAMEORIGIN',
       'x-request-id' : '636D03C6-42FE722901BBC8B5-202ACB0'
       }



#Список с ссылками на категории 
#Решить вопрос с подсчетом количества страниц через try except, отказаться от создания файла с ссылками
 
with open(r'C:\DATA\Data_new.txt', 'w') as file:
    
    for i in range(1):
        for j in range (1):
            file.write(f'https://www.pornhub.com/video?c={i+1}&page={j+1}\n')
            
    file.close()    


data = []
keys = []
i = 0
j = 0
y=0
with open(r'C:\DATA\Data_new.txt', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            req = requests.get(line.strip(), headers = hdr)
            # почему если scr = req.text or src = req.content не работает soup  
            soup = bs(req.text, 'lxml')
            vids = soup.find('ul', class_='nf-videos videos search-video-thumbs').find_all(class_='thumbnail-info-wrapper clearfix')
            vids_time = soup.find('ul', class_='nf-videos videos search-video-thumbs').find_all('div', class_='phimage')
            #сделать запись main_category только раз, после изменения индекса категории
            #main category 
            i += 1
            for vid, vid_t in list(zip(vids, vids_time)):
                try:
                    # продолжительность, что-то придумать со временем загрузки, ссылка на страницу (для видосов с данной страницы)
                    # block = vid.find('div', class_='videoDetailsBlock')
                    duration = vid_t.find('var', class_='duration').text
                    rating = vid.find('div', class_='value').text
                    view_count = view_count_to_num(vid.find('span', class_="views").find('var').text)
                    add_time = vid.find('var', class_="added").text
                    view_key = vid.find('a').get('href')
                    vid_title = vid.find('a')
                    with open(r'C:\DATA\keys.txt', 'a', encoding='utf-8') as text:
                        text.write(f'https://www.pornhub.com{view_key}\n')
                   
                    data.append(
                        {
                            'duration' : duration,
                            'view_count' : view_count, 
                            'rating' : rating,
                            'add_time' : add_time,
                            'view_key' : view_key,
                            'name' : vid_title.text.strip(),
                            
                            }
                        )
                    j+=1
                    print(duration.strip())
                except Exception:
                    print(f'!Problem with video --- {view_key}')
                print(f'{view_count}')
        except Exception:
            print(f'!Problem with category {i} / page {j}! {view_key}')
        print(f'URL  {line} is done!')
            

with open(r'C:\DATA\fulldata.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)
    file.close()
    

    

print(len(data))
# Сделать парсинг по генератору в случае ошибки не парсить все заново, а начать с последней точки 


            
     
       