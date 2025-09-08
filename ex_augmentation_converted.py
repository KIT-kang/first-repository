## ----------------------------------------------------
## 이미지 데이터 증강
## - torchvision.transforms에서 제공하는 기능
## ----------------------------------------------------
## (1) 데이터 준비 및 모듈 로딩
## ----------------------------------------------------
## 모듈 로딩
import matplotlib.pyplot as plt                     # 시각화 모듈
from torchvision import transforms as tf            # 데이터 변형 및 증강
from PIL import Image                               # 파이썬언어기반 이미지 모듈(파이썬 이미지 라이브러리)
import cv2                                          # opencv
import torch

## ----------------------------------------------------
##  이미지 데이터 파일 준비
## ----------------------------------------------------
## 이미지 데이터 폴더
IMG_PATH = '../Data/cat_dog/Cat'

## 이미지 데이터 로딩 타입 : IL.Image
pil_img=Image.open(IMG_PATH)

print(f'shape  : {pil_img.size}')
print(f'width  : {pil_img.width}')
print(f'height : {pil_img.height}')
print(f'mode   : {pil_img.mode}')

## 이미지 데이터 로딩 타입 =>ndarray 타입
img = cv2.imread(IMG_PATH)
print(f'shape : {img.shape}')

## ----------------------------------------------------
##  사용자 정의 함수
## ----------------------------------------------------
## 함수기능 : 이미지 화면 출력
## ----------------------------------------------------
def displayImage(tensor_img, title):
    plt.subplot(1,2, 1)
    plt.imshow(pil_img)
    plt.title("ORIGINAL")

    # 테넛 이미지 C, H, W 순서서
    plt.subplot(1,2,2)
    plt.imshow(tensor_img.permute(1,2,0))
    plt.title(title)

    plt.show()

## ----------------------------------------------------
## 함수기능 : 이미지 정보 출력 함수
## ----------------------------------------------------
def displayInfo(img):
    print(f'shape  : {img.shape}')
    print(f'ndim   : {img.ndim}')
    print(f'values : {img.min()} ~ {img.max()}')

## ----------------------------------------------------
## 함수기능 : 변환된 이미지 파일로 저장장
## ----------------------------------------------------
def saveImage(img, imgpath):
    ## 전처리 이미지 저장
    # - Tensor => PILImage변환환
    retImg = tf.ToPILImage()(img)
    retImg.save(SAVE_PATH)

## ----------------------------------------------------
##  이미지 전처리 : Resize
## ----------------------------------------------------
# 변형 객체 생성
aug=tf.Compose([tf.Resize((100,100)),
                tf.ToTensor()])

# 이미지 변형 : transforms에서는 PIL Image객체 또는 Tensor 객체만 처리리
img=aug(pil_img)


# 변형 후 이미지 속성
displayInfo(img)

# Resize & Tensor 변형 후 비교
displayImage(img, "RESIZE")

## 전처리 이미지 저장
SAVE_PATH = '../Data/image/resize_cat.jpg'
saveImage(img, SAVE_PATH)

## ----------------------------------------------------
##  이미지 전처리 : Rotation
## ----------------------------------------------------
#
# 변형 객체 생성
# expand : 회전에 따른 이미지 전체를 담기 위한 크기 변경 여부 
# center(x, y) : 회전 중심
# fill : 회전 이미지 외부 영역 픽셀 채우기 값 
degree = 90, 121
aug=tf.Compose([tf.RandomRotation((degree), expand=True, fill=255),
                tf.ToTensor()])

for i in range(1, 11):
    # 이미지 변형
    img=aug(pil_img)

    # 이미지 저장
    tf.ToPILImage()(img).save(f"cat_02_ratation_{i:02}.jpg")

    # 변형 후 이미지 속성
    displayInfo(img)

    # Resize & Tensor 변형 후 비교
    plt.subplot(2,5,i)
    plt.imshow(img.permute(1,2,0))
    #plt.title()

plt.show()