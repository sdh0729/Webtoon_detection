import numpy as np
from PIL import Image
import os
import glob

Image.MAX_IMAGE_PIXELS = None
'''이미지 받아오기'''
def file_dir(comic_dir, i):
    files = sorted(glob.glob(comic_dir + '/' + str(i) + '화/out'),key=os.path.getctime)
    return files

def get_image(dirname,cnt):
    img = []
    img_gray=[]
    for i in range(cnt):
        img.append(Image.open(dirname +'/Toon_merged_'+str(i+1)+'.jpg'))
        img_gray.append(Image.open(dirname +'/Toon_merged_'+str(i+1)+'.jpg').convert("L"))
    return [img, img_gray]

def img_cut(path,img,gimg,n):
    x = np.array(gimg)
    dif_row = np.where((x != x[:, 0][:, np.newaxis]).any(axis=1))

    dif_row_list = list()
    start = list()
    end = list()

    for idx in dif_row:
        dif_row_list.append(idx)

    start.append(idx[0] - 1)
    end.append(idx[len(idx) - 1])

    for i in range(1, len(idx) - 1):
        if idx[i + 1] != idx[i] + 1:
            end.append(idx[i] + 1)
            start.append(idx[i + 1] - 1)
        else:
            pass
# start, end리스트 정렬
    ss = sorted(start)
    se = sorted(end)

    if len(ss) != len(se):
        se.append(img.height)

    for j in range(len(se) - 1, -1, -1):
        if se[j] - ss[j] < 20 or se[j] - ss[j] == 181:
            ss.pop(j)
            se.pop(j)

    for i in range(len(se)):
        crop=img.crop((0, ss[i], img.width, se[i]))
        crop = crop.resize((400, 400))
        crop.save(path+f'/{n}_{i}.png', 'png')

if __name__ == '__main__':
    title = str(input("웹툰 제목을 입력해 주세요: "))
    comic_dir = "./"+title
    hwa_dir = glob.glob(comic_dir + "/*화")

    for j in range(len(hwa_dir)):
        path=file_dir(comic_dir,j)
        file_list = os.listdir(path[0])
        file_count = len(file_list)
        img_cnt = get_image(path[0],file_count)

        for i in range(len(img_cnt[0])):
            img_cut(path[0],img_cnt[0][i],img_cnt[1][i], i+1)
