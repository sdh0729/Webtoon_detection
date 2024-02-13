import torch.optim as optim
import timm
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
    transforms.Resize((260, 260)),#크면 학습이 안되네
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

# Define the Vision Transformer model
class VisionTransformer(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, patch_size=16, hidden_size=256, num_layers=6, num_heads=8):
        super(VisionTransformer, self).__init__()

        # Patch Embedding layer
        self.patch_embed = nn.Conv2d(input_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # Positional Embedding
        image_size = 224  # Adjust according to your image size
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # Shape: [batch_size, hidden_size, num_patches_h, num_patches_w]
        x = x.flatten(2).transpose(1, 2)  # Flatten patches, Shape: [batch_size, num_patches, hidden_size]

        # Add positional embeddings
        x = torch.cat([x, self.positional_embedding.repeat(x.size(0), 1, 1)], dim=1)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Classification head
        x = x.mean(dim=1)  # Aggregate patches' information
        x = self.fc(x)

        return x

# 모델 초기화 및 GPU로 이동
model = VisionTransformer(input_channels=3, num_classes=2).to(device)# Assuming 3 channels (RGB) and 2 classes
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train(model, train_loader, val_loader, criterion, optimizer, epochs=20):
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