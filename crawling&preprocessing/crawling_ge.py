import os
import re
import requests
from bs4 import BeautifulSoup
from requests import request
from requests.exceptions import HTTPError
from time import sleep
import errno
from fake_useragent import UserAgent
import ssl

url = "https://newtoki.help/webtoon/"
ssl._create_default_https_context = ssl._create_unverified_context
def download(url, params={}, method='GET', retries=3):
    resp = None
    user_agent = UserAgent()
    headers = {'User-Agent': user_agent.random}

    try:
        resp = request(method, url,
                       params=params if method == 'GET' else {},
                       data=params if method == 'POST' else {},
                       headers=headers)
        resp.raise_for_status()
        # 아니면,
        # if resp.status_code != 200:
    except HTTPError as e:
        if 500 <= e.response.status_code:
            if retries > 0:
                sleep(3)
                resp = download(url, params=params,
                                method=method,
                                retries=retries - 1)
            else:
                print('재방문 횟수 초과')
        else:
            print('Request', resp.request.headers)
            print('Response', e.response.headers)

    return resp
def get_comic_date(en,gen,name):
    date_url = "https://newtoki.help/웹툰/"+en+"/장르/"+gen+"?sst=wr_datetime"

    resp = download(date_url)
    dom = BeautifulSoup(resp.text, 'lxml')
    ddom = dom.select('.box')

    a = list()
    b = list()
    for i in range(len(ddom)):
        dddom = ddom[i].get_attribute_list('onclick')
        tnum = re.findall(r"location.href='https://newtoki.help/webtoon/([0-9]{1,4})", dddom[0])
        tname = list(re.findall('<div class="title"><strong>(.+)</strong>', str(ddom[i])))

        a.append(tnum)
        b.append(tname)
    dic = list(map(list.__add__, b, a))
    found_value = None

    for sublist in dic:
        if sublist[0] == name:
            found_value = sublist[1]
            break

    return found_value
def get_comic_num(x):
    turl = 'https://newtoki.help/webtoon/'+x
    tresp = download(turl)
    p = list(re.findall(r'https://newtoki.help/webtoon/'+x+'/([0-9]{1,6})', tresp.text))
    p.reverse()
    return p
def comic_download(c,r,num,title): # c는 코믹id, r은 회차 id, num은 회차 title은 제목

    resp = download(url + c + '/' + r)
    dom = BeautifulSoup(resp.text, 'lxml')

    target_div = dom.find('div', {'id': 'bo_v_con'})
    if target_div:
        src_list = [tag.get('src') for tag in target_div.find_all(attrs={'src': True})]

    save_path = title+'/'+str(num)+'화/Toon_%s'
    length = len(src_list) - 6

    for i in range(0, length):
        response = requests.get(src_list[5 + i])
        response.raise_for_status()
        with open(save_path % (i) + '.jpg', 'wb') as file:
            file.write(response.content)

if __name__ == "__main__":

    print("***뉴토끼 다운로더***\n")
    ff = str(input("완결 웹툰이면 '완결'을, 아니면 '일반'을 입력하세요. : "))
    ge = str(input("장르를 입력하세요. : "))
    title = str(input("웹툰 제목을 입력해 주세요: "))

    c_id = get_comic_date(ff, ge, title)#4133
    r_id = get_comic_num(c_id)

    try:
        if not (os.path.isdir(title)):
            os.makedirs(os.path.join(title))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
        
    for i in range(len(r_id)):
        os.makedirs(os.path.join(title+'/'+str(i)+'화'))
        comic_download(c_id,r_id[i],i,title)
        print(f'{i}화 완료')
    print('끝')


