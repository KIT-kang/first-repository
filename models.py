## -------------------------------------------------------------
## 사용자가 정의한 모델 클래스 및 데이터셋 클래스들
## -------------------------------------------------------------

## -------------------------------------------------------------
## 모듈 로딩
## -------------------------------------------------------------
import torch
import torch.nn as nn

## -------------------------------------------------------------
## 커스텀 클래스 모델 정의 : 피쳐 784개, 타겟 10가지 0 ~ 9
##              입력    출력    활성함수
## 입력층        794    255     ReLU
## 은닉층        255    100     ReLU
## 출력층        50     10      Softmax => 손실함수에 따라 정함
## -------------------------------------------------------------   
class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 255),
            nn.ReLU(),
            nn.Linear(255, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        return self.net(x)
    
## -------------------------------------------------------------
## 커스텀 클래스 모델 정의 : 피쳐 784개, 타겟 10가지 0 ~ 9
## 클래스 이름 : MnistClassifierCNN
##                      입력    출력    패딩    데이터사이즈
## 특징추출부 CONV2D      1       32     1     28x28x1=>28x28x32
##          POOL2D                           28x28x32=>14x14x32
##          CONV2D     32       64     1     14x14x32=>14x14x64
##          POOL2D                           14x14x64=>7x7x64
## 학습진행부 FLATTEN                           7x7x64=>3,136
##          LINEAR      3136    128     x
##          LINEAR      128     10
## -------------------------------------------------------------    
class MnistClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ## 이미지 특징 추출 부분분
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 14x14 -> 7x7
            # 특징 학습(분류) 부분
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    # 전방향 학습
    def forward(self, x):
        # CONV2D 층 입력 형태 : 4D or 3D
        # - 4D :(배치사이즈, 이미지채널, 행, 열)
        # 입력 2D => 4D 변형
        x = x.view(-1, 1, 28, 28)  # (B, 784) → (B, 1, 28, 28)
        return self.net(x)