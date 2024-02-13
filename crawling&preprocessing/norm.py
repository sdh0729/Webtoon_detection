import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms

def calculate_norm(dataset): # 이미지 데이터 정규화에 사용할 평균 표준편차 값을 구하는 함수
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

transform = transforms.Compose([transforms.ToTensor(),])

cool = torchvision.datasets.ImageFolder(root='dataset', transform=transform)

mean_, std_ = calculate_norm(cool)

print(f'평균(R,G,B): {mean_}\n표준편차(R,G,B): {std_}')