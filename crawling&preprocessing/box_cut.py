from PIL import Image
import json
import os

def crop_images_from_json(json_file_path, image_folder_path, output_folders):
    # JSON 파일에서 바운딩 박스 정보를 읽어옴
    with open(json_file_path, 'r') as json_file:
        annotations = json.load(json_file)

    # 각 이미지에 대해 작업 수행
    for key, data in annotations.items():
        filename = data['filename']
        image_path = os.path.join(image_folder_path, filename)
        if os.path.isfile(image_path):
            try:
                with Image.open(image_path) as img:
                    regions = data.get('regions', {})
                    for region_id, region_data in regions.items():
                        shape_attributes = region_data.get('shape_attributes', {})
                        x = shape_attributes.get('x', 0)
                        y = shape_attributes.get('y', 0)
                        width = shape_attributes.get('width', 0)
                        height = shape_attributes.get('height', 0)
                        region_attributes = region_data.get('region_attributes', {})
                        if 'text' in region_attributes:
                            box_type = 'text'
                        elif 'person' in region_attributes:
                            box_type = 'person'
                        else:
                            continue  # 'text' 또는 'person' 레이블이 없으면 처리 건너뜀

                        # 이미지를 박스에 맞게 자르기
                        cropped_img = img.crop((x, y, x + width, y + height))

                        # 새로운 이미지를 저장할 폴더 생성
                        if box_type in output_folders:
                            output_folder = output_folders[box_type]
                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

                            # 이미지를 새로운 파일로 저장
                            output_path = os.path.join(output_folder, f"{key}_{region_id}.jpg")
                            cropped_img.save(output_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 사용 예시
json_file_path = 'anot.json'
image_folder_path = 'imgs'

# 각 박스 종류에 따른 결과 저장 폴더 설정
output_folders = {
    'text': 'box_cut/text',
    'person': 'box_cut/person'
}

crop_images_from_json(json_file_path, image_folder_path, output_folders)
