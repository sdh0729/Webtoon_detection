import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 데이터 로드와 전처리
# 여기에 데이터셋과 DataLoader를 설정해주세요.
# transform은 선정적인 이미지와 그렇지 않은 이미지에 맞게 조정되어야 합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((300, 300)),
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

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 입력 이미지의 크기를 기반으로 FC 레이어의 입력 크기 계산
        self.fc_input_size = self._get_fc_input_size(input_size)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def _get_fc_input_size(self, input_size):
        # 모델에 이미지를 한 번 전달하여 FC 레이어의 입력 크기 계산
        with torch.no_grad():
            x = torch.zeros(1, *input_size)
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, self.fc_input_size)
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 모델 초기화 및 손실 함수, 옵티마이저 설정
input_size = (3, 300, 300)  # (채널 수, 이미지 높이, 이미지 너비)
model = CNN(input_size).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 및 평가 과정
num_epochs = 20
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_total = 0
    val_total = 0
    train_correct = 0
    val_correct = 0

    # Training loop
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.squeeze(1)  # 모델의 출력에서 차원을 조정합니다.
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        predicted = torch.round(outputs)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_losses.append(train_loss / len(train_loader))
    train_accs.append(train_correct / train_total)

    # Validation loop
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.squeeze(1)  # 모델의 출력에서 차원을 조정합니다.
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()

            predicted = torch.round(outputs)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_correct / val_total)

# Loss 및 Accuracy 그래프 표시
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Test loop
test_loss = 0.0
test_total = 0
test_correct = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        outputs = outputs.squeeze(1)  # 모델의 출력에서 차원을 조정합니다.
        loss = criterion(outputs, labels.float())
        test_loss += loss.item()

        predicted = torch.round(outputs)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = test_correct / test_total
test_loss = test_loss / len(test_loader)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

