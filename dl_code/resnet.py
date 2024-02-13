import torch.optim as optim
from torchvision import models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# GPU 사용 여부 설정
device = torch.device("mps")

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),#resnet이 사용
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.678, 0.608, 0.622], std=[0.238, 0.247, 0.242]),
])

# 데이터셋 불러오기
train_dataset = datasets.ImageFolder('output_data/train', transform=transform)
val_dataset = datasets.ImageFolder('output_data/val', transform=transform)
test_dataset = datasets.ImageFolder('output_data/test', transform=transform)

# 각각의 DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class BinaryImageClassifier(nn.Module):
    def __init__(self):
        super(BinaryImageClassifier, self).__init__()
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')  # 미리 학습된 ResNet 모델 로드
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 이진 분류기의 출력은 2
        )
        self.resnet = resnet

    def forward(self, x):
        x = self.resnet(x)
        return x

# 모델 초기화 및 GPU로 이동
model = BinaryImageClassifier().to(device)

# 손실 함수와 최적화 기법 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수 정의
def train(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 GPU로 이동
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        train_losses.append(running_loss / len(train_loader))
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch + 1} - Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation 과정
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Loss 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy 그래프 그리기
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 데이터를 GPU로 이동
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

# 훈련 실행
train(model, train_loader, val_loader, criterion, optimizer, epochs=20)

# Test 데이터로 모델 평가
test_loss, test_accuracy = validate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")