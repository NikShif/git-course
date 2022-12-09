import requests
from bs4 import BeautifulSoup
import csv
import json
import numpy as np 
import os

#Список с ссылками на категории 
 
with open(r'C:\DATA\Data.txt', 'w') as file:
    for i in range(108):
        file.write(f'https://www.pornhub.com/video?c={i+1}\n')
    file.close()    

with open(r'C:\DATA\Data.txt', 'r') as file:
    for line in file:
        req = requests.get(line)
# почему если scr = req.text or src = req.content не работает soup  
        soup = BeautifulSoup(req.text, 'lxml')
        
        
        
        
    

     
class PornVideoData:
    
    try:
        os.mkdir('C://DATA')
    except Exception:
        print("Файлы будут записаны в другую папку! \n ") 
        
    with open(r'C:\DATA\Data.json', 'w') as file:
        pass
        
    def __init__(self, veikeys: list, path_saving_file: str,
                  folder_name):
        self.__veikeys = veikeys
        self.__path_saving_file = path_saving_file
        
        
        pass

















class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height
    def get_width(self):
        return self._width
    def s_width(self, w):
        self._width = w
    def g_height(self):
        return self._height
    def set_height(self, h):
        self._height = h