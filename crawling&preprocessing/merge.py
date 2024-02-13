import glob
import os
from PIL import Image, ImageFile
import errno

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def combineImage(cname, rname, full_width, full_height, image_key, image_list, index, max_height):
    canvas = Image.new('RGB', (full_width, full_height), 'white')
    output_height = 0

    for im in image_list:
        width, height = im.size
        canvas.paste(im, (0, output_height))
        output_height += height

    # 이미지 높이가 최대 높이(max_height)를 초과하면 조정하여 저장
    if full_height > max_height:
        ratio = max_height / full_height
        new_width = int(full_width * ratio)
        new_height = int(full_height * ratio)
        canvas = canvas.resize((new_width, new_height), Image.ANTIALIAS)

    canvas.save(cname + '/' + rname + '/out/' + image_key + '_merged_' + str(index) + '.jpg')


def listImage(cname, rname, image_key, image_value, max_height):
    full_width, full_height, index = 0, 0, 1
    image_list = []

    for i in image_value:
        im = Image.open(cname + '/' + rname + '/' + image_key + "_" + str(i) + ".jpg")
        width, height = im.size

        if full_height + height > max_height:
            combineImage(cname, rname, full_width, full_height, image_key, image_list, index, max_height)
            index += 1
            image_list = []
            full_width, full_height = 0, 0

        image_list.append(im)
        full_width = max(full_width, width)
        full_height += height

    combineImage(cname, rname, full_width, full_height, image_key, image_list, index, max_height)


def file_dir(comic_dir, i):
    files = sorted(glob.glob(comic_dir + '/' + str(i) + '화/*.jpg'), key=os.path.getctime)
    r_list = []
    for f in files:
        cname = f.split('/')[1]
        rname = f.split('/')[2]
        name = f.split('/')[3]
        key = name.split('_')[0]
        value = name.split('_')[1].split('.')[0]
        r_list.append(value)
    return [[cname, rname, name, key], r_list]


if __name__ == '__main__':
    print('[+] Start Combining Images')
    title = str(input("웹툰 제목을 입력해 주세요: "))
    comic_dir = "./" + title
    hwa_dir = glob.glob(comic_dir + "/*")

    # Make Directory
    for i in range(len(hwa_dir)):
        try:
            if not (os.path.isdir(comic_dir + '/' + str(i) + '화/out')):
                os.makedirs(os.path.join(comic_dir + '/' + str(i) + '화/out'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise

    max_height = 4000  # 이미지 높이 최댓값

    for i in range(len(hwa_dir)):
        a = file_dir(comic_dir, i)
        listImage(a[0][0], a[0][1], a[0][3], a[1], max_height)

    print('[+] Complete Combining Images')
