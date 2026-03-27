import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ── 디바이스 ───────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 데이터 ────────────────────────────────────────────────
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

train_loader = DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
    batch_size=256,
    shuffle=True,
    num_workers=0,         
    pin_memory=True
)

test_loader = DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform_test),
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ── 모델 ─────────────────────────────────────────────────
class FastCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── 학습 설정 ─────────────────────────────────────────────
model = FastCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCHS = 30

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)

# 🔥 AMP (속도 + 안정성)
scaler = torch.cuda.amp.GradScaler()

# ── 학습 ─────────────────────────────────────────────────
def train():
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # 🔥 mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100. * correct / total

# ── 평가 ─────────────────────────────────────────────────
def evaluate():
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total

# ── 실행 ─────────────────────────────────────────────────
print(f"\n{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>10} {'Test Acc':>10} {'Time':>7}")
print("-" * 55)

best_acc = 0.0
start = time.time()

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = train()
    test_acc = evaluate()

    elapsed = time.time() - t0

    print(f"{epoch:>5} {train_loss:>10.4f} {train_acc:>9.2f}% {test_acc:>9.2f}% {elapsed:>6.1f}s")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✅ Best model saved! ({best_acc:.2f}%)")

print(f"\n총 소요 시간 : {time.time()-start:.1f}s")
print(f"최고 정확도  : {best_acc:.2f}%")
