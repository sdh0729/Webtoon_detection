from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import os
import errno
import requests
from selenium.webdriver.support.ui import WebDriverWait
import time
import urllib
import os
import urllib.request
import re

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=chrome_options)
driver.get("https://blacktoon267.com/")
wait = WebDriverWait(driver, 10)
time.sleep(0.5)
def get_num(ff,ctitle):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # 페이지를 맨 아래로 스크롤합니다.
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 스크롤 기다림
        time.sleep(0.5)  # 적절한 시간 대기 (필요에 따라 조정)

        # 현재 스크롤 높이를 가져옵니다.
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # 더 이상 스크롤이 이동하지 않으면 반복문 종료
        last_height = new_height

    driver.find_element(By.XPATH, ff).click()  #클릭
    time.sleep(1)
    html = driver.page_source
    soup = BeautifulSoup(html,'html.parser')
    ellements = soup.find_all('div', {'class': 'col-4 col-sm-4 col-md-2'}) # 보이는거 만화 코드
    elements = soup.find_all('div', {'class': 'col-4 col-sm-4 col-md-2 hide'})# 숨겨진거 만화 코드
    c_id = []
    c_title = []
    # 보이는 거 ID 값 추출하기
    for element in ellements:
        element_id = element.get('id').split('_')[1]
        c_id.append(element_id)
    #안 보이는 거 ID 값 추출하기
    for element in elements:
        element_id = element.get('id').split('_')[1]
        c_id.append(element_id)

    #Finding all elements with the class 'book-del delete_1'
    div_elements = soup.find_all('div', {'class': 'book-del delete_1'})
    #Extracting and printing the 'title' attribute from 'toon-link' elements within 'book-del delete_1' divs
    for div_element in div_elements:
        # Finding all 'a' elements with the class 'toon-link' within each 'book-del delete_1' div
        toon_links = div_element.find_all('a', {'class': 'toon-link'})
        # Extracting 'title' attribute from each 'toon-link' element
        for link in toon_links:
            title = link.get('title')
            c_title.append(title)
    matching = list(map(list, zip(c_title, c_id)))

    found_value = None

    for sublist in matching:
        if sublist[0] == ctitle:
            found_value = sublist[1]
            break

    return found_value
def get_total_cnt(comicnum):
    click_comic = '// *[ @ id = "toon_' + str(comicnum) + '"] / div[1] / a'

    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # 페이지를 맨 아래로 스크롤합니다.
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 스크롤 기다림
        time.sleep(0.5)  # 적절한 시간 대기 (필요에 따라 조정)

        # 현재 스크롤 높이를 가져옵니다.
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # 더 이상 스크롤이 이동하지 않으면 반복문 종료
        last_height = new_height

    driver.find_element(By.XPATH, click_comic).click()
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    div_cards = soup.find_all('div', {'class':'card card-fluid mt-2 toon-link-s'})
    div_cards_hide = soup.find_all('div',{'class':'card card-fluid mt-2 toon-link-s hide'})
    num = []

    for div_card in div_cards:
        anchor_tag = div_card.find('a')
        if anchor_tag:
            attr_id_value = anchor_tag.get('attr-id')
            if attr_id_value:
                num.append(attr_id_value)

    for div_card in div_cards_hide:
        anchor_tag = div_card.find('a')
        if anchor_tag:
            attr_id_value = anchor_tag.get('attr-id')
            if attr_id_value:
                num.append(attr_id_value)
    num.reverse()
    return num

def comic_download(click_hwa,hwa,title):
    img_src = []
    click_comic = '//*[@id="toon_list"]/div['+str(click_hwa)+']/a'
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # 페이지를 맨 아래로 스크롤합니다.
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 스크롤 기다림
        time.sleep(0.5)  # 적절한 시간 대기 (필요에 따라 조정)

        # 현재 스크롤 높이를 가져옵니다.
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # 더 이상 스크롤이 이동하지 않으면 반복문 종료
        last_height = new_height
    time.sleep(1)
    driver.find_element(By.XPATH, click_comic).click()
    time.sleep(1)

    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # 페이지를 맨 아래로 스크롤합니다.
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 스크롤 기다림
        time.sleep(2)  # 적절한 시간 대기 (필요에 따라 조정)

        # 현재 스크롤 높이를 가져옵니다.
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # 더 이상 스크롤이 이동하지 않으면 반복문 종료
        last_height = new_height
    time.sleep(4)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    # Find the div with id 'toon_content_imgs'
    toon_content_div = soup.find('div', {'id': 'toon_content_imgs'})

    #Find all img tags within toon_content_div and download their src content

    if toon_content_div:
        img_tags = toon_content_div.find_all('img')
        img_src = [img_tag.get('src') for img_tag in img_tags if img_tag.get('src')]  # Filter out None values

        # Create directory if it doesn't exist
        directory = f"{title}/{str(hwa)}화"
        os.makedirs(directory, exist_ok=True)

        # Add headers to mimic a browser visit
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'
        }

        # Assuming title and hwa are defined earlier
        for i, img_url in enumerate(img_src):
            print(img_url)  # Assuming the src needs the base URL
            try:
                img_data = requests.get(img_url, headers=headers).content
                file_path = os.path.join(directory, f"Toon_{i}.jpg")
                with open(file_path, 'wb') as img_file:
                    img_file.write(img_data)
            except Exception as e:
                print(f"Failed to fetch image {img_url}: {e}")

    # if toon_content_div:
    #     img_tags = toon_content_div.find_all('img')
    #     img_src = [img_tag.get('src') for img_tag in img_tags if img_tag.get('src')]  # Filter out None values
    #
    #     # Create directory if it doesn't exist
    #     directory = f"{title}/{str(hwa)}화"
    #     os.makedirs(directory, exist_ok=True)
    #
    #     for i, img_url in enumerate(img_src):
    #         print(img_url)  # Assuming the src needs the base URL
    #         try:
    #             file_path = os.path.join(directory, f"Toon_{i}.jpg")
    #             urllib.request.urlretrieve(img_url, file_path)
    #         except Exception as e:
    #             print(f"Failed to fetch image {img_url}: {e}")

    driver.back()
    driver.back()
    time.sleep(1)

if __name__ == "__main__":
    url = 'https://blacktoon266.com/webtoon/'

    ended = r'//*[@id="profile-tab"]' #완결 탭
    mon = '// *[ @ id = "home"] / div / button[2]'
    tue = '// *[ @ id = "home"] / div / button[3]'
    wed = '// *[ @ id = "home"] / div / button[4]'
    thu = '// *[ @ id = "home"] / div / button[5]'
    fri = '// *[ @ id = "home"] / div / button[6]'
    sat = '// *[ @ id = "home"] / div / button[7]'
    sun = '// *[ @ id = "home"] / div / button[8]'
    ten = '// *[ @ id = "home"] / div / button[9]'

    ff = str(input("완결 웹툰이면 '완결'을, 아니면 '요일'을 입력하세요. : "))

    if ff=='완결':
        ff = ended
    elif ff=='월':
        ff =mon
    elif ff=='화':
        ff = tue
    elif ff=='수':
        ff = wed
    elif ff=='목':
        ff = thu
    elif ff=='금':
        ff = fri
    elif ff=='토':
        ff = sat
    elif ff=='일':
        ff = sun
    elif ff=='열흘':
        ff = ten

    title = str(input("웹툰 제목을 입력해 주세요: "))

    try:
        if not (os.path.isdir(title)):
            os.makedirs(os.path.join(title))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    comic_num=get_num(ff,title)
    time.sleep(2)
    num = get_total_cnt(comic_num)
    for i in range(len(num)):
        try:
            if not (os.path.isdir(title + '/' + str(i) + '화')):
                os.makedirs(os.path.join(title + '/' + str(i) + '화'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise

        comic_download(len(num)-i,i,title)
        time.sleep(3)
    driver.quit()
