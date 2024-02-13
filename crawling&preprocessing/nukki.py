import cv2
import numpy as np
# 모델 불러오기 (사람을 감지하는 YOLO 모델)
net = cv2.dnn.readNet('yolov3.weights',"yolov3.cfg")  # 가중치 파일과 설정 파일 경로

# 클래스 이름 불러오기
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# 이미지 불러오기
image = cv2.imread('dataset/cool/1_1_2_3.png')
height, width = image.shape[:2]

# 이미지를 모델 입력 사이즈에 맞게 전처리
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# 네트워크에 이미지 입력
net.setInput(blob)

# 출력층 가져오기
output_layers = net.getUnconnectedOutLayersNames()

# 예측 수행
layer_outputs = net.forward(output_layers)

# 감지된 객체 정보 저장
boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and class_id == 0:  # 사람 클래스
            box = detection[:4] * np.array([width, height, width, height])
            (center_x, center_y, w, h) = box.astype("int")

            x = int(center_x - (w / 2))
            y = int(center_y - (h / 2))

            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 비최대 억제
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 윤곽선 그리기
if len(indices) > 0:
    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 결과 보여주기
    cv2.imshow("Detected People", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No objects detected.")

