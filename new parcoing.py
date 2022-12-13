from datetime import time, datetime
import requests
from bs4 import BeautifulSoup as bs
import csv
import json
import numpy as np 
import os
import re

   
hdr = {
       'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
       'accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
       'x-frame-options' : 'SAMEORIGIN',
       'x-request-id' : '636D03C6-42FE722901BBC8B5-202ACB0'
       }


req = requests.get('https://www.pornhub.com/view_video.php?viewkey=ph627e1d99f018a', headers=hdr)
soup = bs(req.text, 'lxml')
vids = soup.find('script', type='application/ld+json')


