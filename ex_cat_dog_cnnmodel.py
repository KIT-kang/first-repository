## -------------------------------------------------------------
## 목표 : 고양이와 개 이미지 분류 CNN 모델 개발 (데이터셋 포함)
## -------------------------------------------------------------

## -------------------------------------------------------------
## 모듈 로딩
## -------------------------------------------------------------
## - 일반적ㅇ니 파이토치 인공신경망 관련 모듈듈
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

## - 이미지 전처리 및 데이터셋 관련 모듈듈
from torchvision.datasets import ImageFolder
from torchvision import transforms
## - 시각화 모듈듈
import matplotlib.pyplot as plt

## -------------------------------------------------------------
## 데이터 준비 및 전처리 구성
## -------------------------------------------------------------
## - 데이터 준비
IMG_ROOT = '../Data/cat_dog/'

## - 이미지 데이터 전처리 
preprocessing = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])

## -------------------------------------------------------------
## 데이터 셋 객체 생성
## -------------------------------------------------------------
## - 데이터셋 객체 생성
imgDS = ImageFolder(root=IMG_ROOT, transform=preprocessing)

## - 분류 라벨
IDX_TO_CLASS = {v: k for k, v in imgDS.class_to_idx.items()}

## - 체크
print(f'IDX_TO_CLASS => {IDX_TO_CLASS}')
print(f'imgDataset 개수 : {len(imgDS.targets)}개')
print(f'imgDataset 분류 : {imgDS.class_to_idx}')
print(f'- cat 개수 : {imgDS.targets.count(0)}개, {(imgDS.targets.count(0)/len(imgDS.targets))*100:.2f}%')
print(f'- dog 개수 : {imgDS.targets.count(1)}개, {(imgDS.targets.count(1)/len(imgDS.targets))*100:.2f}%')

## -------------------------------------------------------------
## 데이터 분할 및 로더 구성 (8:1:1)
## -------------------------------------------------------------
dataset_size = len(imgDS)
train_size = int(dataset_size * 0.8)
valid_size = int(dataset_size * 0.1)
test_size  = dataset_size - train_size - valid_size

## - 학습, 검증, 테스트용 데이터셋 생성
trainDS, validDS, testDS = random_split(imgDS, 
                                        [train_size, valid_size, test_size], 
                                        generator=torch.Generator().manual_seed(42))

## -------------------------------------------------------------
## CNN 모델 정의
## -------------------------------------------------------------
class CatDogClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ## - 이미지의 특징 추출
            ## - 입력 : 컬러이미지 3, 출력 : 커널 3x3 32개, SAME : 입력 H, W == 출력 H, W
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 32, 100, 100 => 32, 50, 50

            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 23, 50, 50 => 64, 50, 50
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 64, 50, 50 => 64, 25, 25

            # - 분류 학습 부분
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 128),                   # 이미지 주요 특징 1D. 누런갯수
            nn.ReLU(),
            nn.Linear(128, 2)                               # 뉴런출력수 128, 분류클래스
        )

    def forward(self, x):
        return self.net(x)

## -------------------------------------------------------------
## 학습 함수
## -------------------------------------------------------------
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        # 학습
        optimizer.zero_grad()
        outputs = model(x_batch)

        # 손실 계산 및 역전파 & 업데이트
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # 배치 단위의 손실 누적
        total_loss += loss.item()
        print(".", end="")
    # 1에포크에 대ㅏㄴ 평균 손실 값
    return total_loss / len(dataloader)

## -------------------------------------------------------------
## 평가 함수
## -------------------------------------------------------------
def evaluate(model, dataloader, criterion):
    # 평가 모드 설정
    model.eval()

    # 평가 전체 손실, 정답개수, 전체 데이터수수
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            # 예측
            outputs = model(x_batch)

            # 손실 계산
            loss = criterion(outputs, y_batch)

            #배치단위 손실 누적
            total_loss += loss.item()

            # 예측 클래스 추출
            preds = outputs.argmax(dim=1)

            # 예측과 정답 동일여부 카운팅
            correct += (preds == y_batch).sum().item()

            total += y_batch.size(0)
    # 정확도 계산 및 평균 손실값 계산
    accuracy = correct / total
    avg_loss = total_loss /len(dataloader)
    return avg_loss, accuracy
## -------------------------------------------------------------
## 예측 함수
## -------------------------------------------------------------
def predict(model, data):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        predictions = outputs.argmax(dim=1)
    return predictions

## -------------------------------------------------------------
## 학습 준비 및 실행
## -------------------------------------------------------------
## 학습 관련 변수 설정
_LR     = 0.001
EPOCHS  = 20

## 데이터로더 객체 생성
trainDL = DataLoader(trainDS, batch_size=32, shuffle=True)
validDL = DataLoader(validDS, batch_size=32, shuffle=False)
testDL  = DataLoader(testDS, batch_size=32, shuffle=False)

## 학습 관련 객체 생성
model       = CatDogClassifierCNN()
criterion   = nn.CrossEntropyLoss()
optimizer   = optim.Adam(model.parameters(), lr=_LR)

## 학습 진행
BEST_ACC = 0.0
print("START LERANING")
for epoch in range(1, EPOCHS+1):
    train_loss = train(model, trainDL, criterion, optimizer)
    val_loss, val_acc = evaluate(model, validDL, criterion)

    ## 학습 진행도 출력
    #if epoch % 5 == 0 or epoch == 1:
    print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    ## 모델 가중치 저장
    if val_acc > BEST_ACC:
        torch.save(model.state_dict(), "cat_dog_cnn_weights.pt")
        BEST_ACC = val_acc

## -------------------------------------------------------------
## 테스트
## -------------------------------------------------------------
correct, total = 0, 0
for x_batch, y_batch in testDL:
    y_pre = predict(model, x_batch)
    correct += (y_pre == y_batch).sum().item()
    total += y_batch.size(0)
print(f"[CatDog] Test Accuracy: {correct / total:.4f}")
