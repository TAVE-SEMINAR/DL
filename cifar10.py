import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# =========================================================
# 1. 디바이스
# =========================================================
device = torch.device("cpu")

# =========================================================
# 2. 모델 정의
# =========================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# 3. 모델 생성
# =========================================================
model = MLP().to(device)

# =========================================================
# 4. 학습 (이미 저장된 모델 있으면 skip)
# =========================================================
import os

if not os.path.exists("mnist_mlp.pth"):
    print("학습 시작...")

    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/3] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "mnist_mlp.pth")
    print("모델 저장 완료")

# =========================================================
# 5. 모델 로드
# =========================================================
model.load_state_dict(torch.load("mnist_mlp.pth", map_location=device))
model.eval()

# =========================================================
# 6. FastAPI
# =========================================================
app = FastAPI()

# 전처리
transform_api = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    image = transform_api(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    return {"prediction": pred}
