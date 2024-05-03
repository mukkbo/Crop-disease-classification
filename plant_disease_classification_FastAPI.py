import io
import cv2
import numpy as np
from cv2.dnn import Model
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
import torch
from torch.nn import functional as F
from torchvision.models import resnet50
import torch.nn as nn
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from timm import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Request body에 대한 데이터 모델 정의
class Item(BaseModel):
    image_path: str
    crop: str

# 작물별 설정값을 담은 클래스
class cfg:
    input_size1 = (224, 224)
    input_size2 = (256, 256)

    ddalki_num_classes = 7
    ddalki_class_list = ['00', 'a1', 'a2', 'b1', 'b6', 'b7', 'b8']
    ddalki_class_list_name = ['정상', '딸기잿빛곰팡이병', '딸기흰가루병', '냉해피해', '다량원소결핍(N)', '다량원소결핍(P)', '다량원소결핍(K)']

    tomato_num_classes = 8
    tomato_class_list = ['00', 'a5', 'a6', 'b2', 'b3', 'b6', 'b7', 'b8']
    tomato_class_list_name = ['정상', '토마토흰가루병', '토마토잿빛곰팡이병', '열과','칼슘결핍', '다량원소결핍(N)', '다량원소결핍(P)', '다량원소결핍(K)']

    paprica_num_classes = 7
    paprica_class_list = ['00', 'a9', 'a10', 'b3', 'b6', 'b7', 'b8']
    paprica_class_list_name = ['정상', '파프리카흰가루병', '파프리카잘록병', '칼슘결핍', '다량원소결핍(N)', '다량원소결핍(P)', '다량원소결핍(K)']

    oe_num_classes = 7
    oe_class_list = ['00', 'a3', 'a4', 'b1', 'b6', 'b7', 'b8']
    oe_class_list_name = ['정상', '오이노균병', '오이흰가루병', '냉해피해', '다량원소결핍(N)', '다량원소결핍(P)', '다량원소결핍(K)']

    gochu_num_classes = 7
    gochu_class_list = ['00', 'a7', 'a8', 'b3', 'b6', 'b7', 'b8']
    gochu_class_list_name = ['정상', '고추탄저병', '고추흰가루병', '칼슙결핍', '다량원소결핍(N)', '다량원소결핍(P)', '다량원소결핍(K)']

    podo_num_classes = 5
    podo_class_list = ['00', 'a11', 'a12', 'b4', 'b5']
    podo_class_list_name = ['정상', '탄저병', '노균병', '일소피해', '축과병']


app = FastAPI()

class ViT_MAE(nn.Module):
    def __init__(self, num_classes, model_name='vit_base_patch16_224.mae', pretrained=False):
        super(ViT_MAE, self).__init__()
        self.model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x), None

class ViT(nn.Module):
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=False):
        super(ViT, self).__init__()
        self.model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet50, self).__init__()
        self.base_model = resnet50(pretrained=pretrained)
        self.base_model.fc = nn.Identity()
        self.mid1 = nn.Linear(2048, 1024)
        self.mid2 = nn.Linear(1024, 512)
        self.mid3 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, num_classes)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.base_model(x)
        x = nn.ReLU()(self.mid1(x))
        x = nn.ReLU()(self.mid2(x))
        x = nn.ReLU()(self.mid3(x))
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2

def load_model(crop_name):
    # crop_name에 따라 모델 로드
    # cfg_crop 내의 각 작물별 클래스 수는 적절히 정의되어 있어야 함
    # 모델의 저장 경로는 실제 파일 경로로 변경해야 함 /model/6_best_max_acc_v2.pt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if crop_name == '딸기':
        model = ResNet50(pretrained=False, num_classes=cfg.tomato_num_classes)
        model.load_state_dict(torch.load('./model/1_best_max_acc_v2.pt', map_location=device)['model_state_dict'])
        num = 0
    elif crop_name == '토마토':
        model = ResNet50(pretrained=False, num_classes=cfg.tomato_num_classes)
        model.load_state_dict(torch.load('./model/2_best_max_acc_v2.pt', map_location=device)['model_state_dict'])
        num = 0
    elif crop_name == '파프리카':
        model = ViT(pretrained=False, num_classes=cfg.paprica_num_classes)
        model.load_state_dict(torch.load('./model/3_best_max_acc-V2.pt', map_location=device)['model_state_dict'])
        num = 1
    elif crop_name == '오이':
        model = ResNet50(pretrained=False, num_classes=cfg.oe_num_classes)
        model.load_state_dict(torch.load('./model/4_best_max_acc_v2.pt', map_location=device)['model_state_dict'])
        num = 0
    elif crop_name == '고추':
        model = ResNet50(pretrained=False, num_classes=cfg.gochu_num_classes)
        model.load_state_dict(torch.load('./model/5_best_max_acc_v2.pt', map_location=device)['model_state_dict'])
        num = 0
    elif crop_name == '포도':
        model = ViT(pretrained=False, num_classes=cfg.podo_num_classes)
        model.load_state_dict(torch.load('./model/6_best_max_acc-V2.pt', map_location=device)['model_state_dict'])
        num = 1
    return model, num

def transform(img, num):
    #이미지 전처리 메서드
    if num == 1:
        input_size = (224, 224)
    else :
        input_size = (256, 256)
    img = cv2.resize(img, dsize=input_size)

    base_augs = [
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0),
        ToTensorV2()
    ]
    transformed = A.Compose(base_augs)
    img = transformed(image=img)['image']
    return img


def preprocess_image_new(image_path,num):
    if num == 1:
        input_size = (224, 224)
    else:
        input_size = (256, 256)
    image = cv2.imread(image_path)
    image_copy = Image.open(image_path)
    input_data = transform(image,num)
    input_data = input_data.unsqueeze(0)
    return input_data, image_copy

async def preprocess_image_file_new(file: UploadFile, num):
    # 이미지 파일을 바이트 데이터로 읽어들입니다.
    image_stream = await file.read()
    # 바이트 데이터를 numpy 배열로 변환합니다.
    image_copy = Image.open(io.BytesIO(image_stream))
    image_stream = np.frombuffer(image_stream, np.uint8)
    if num == 1:
        input_size = (224, 224)
    else :
        input_size = (256, 256)
        
    image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
    input_data = transform(image, num)
    input_data = input_data.unsqueeze(0)
    return input_data, image_copy

def select_class_list(crop):
    if crop == '딸기':
        class_list_name = cfg.ddalki_class_list_name
    elif crop == '토마토':
        class_list_name = cfg.tomato_class_list_name
    elif crop == '파프리카':
        class_list_name = cfg.paprica_class_list_name
    elif crop == '오이':
        class_list_name = cfg.oe_class_list_name
    elif crop == '고추':
        class_list_name = cfg.gochu_class_list_name
    elif crop == '포도':
        class_list_name = cfg.podo_class_list_name
    return class_list_name

def predict_by_img_url(path, crop):
    # 모델 로드 및 예측 수행
    # preprocess_image_new와 cfg.podo_class_list_name는 적절히 정의되어 있어야 함
    model, num = load_model(crop)
    model.eval()

    input_data, input_copy = preprocess_image_new(path, num)

    with torch.no_grad():
        output, _ = model(input_data)
        probs = F.softmax(output, dim=1)

    prob_and_disease = [(prob, disease) for prob, disease in zip(probs[0], select_class_list(crop))]
    prob_and_disease.sort(key=lambda x: x[0], reverse=True)


    return prob_and_disease[0][0].item(), prob_and_disease[1][0].item(), prob_and_disease[0][1], prob_and_disease[1][1]

async def predict(img:UploadFile, crop:str):
    # 모델 로드 및 예측 수행
    # preprocess_image_new와 cfg.podo_class_list_name는 적절히 정의되어 있어야 함
    model, num = load_model(crop)
    model.eval()

    input_data, input_copy = await preprocess_image_file_new(img, num)

    with torch.no_grad():
        output, _ = model(input_data)
        probs = F.softmax(output, dim=1)

    prob_and_disease = [(prob, disease) for prob, disease in zip(probs[0], select_class_list(crop))]
    prob_and_disease.sort(key=lambda x: x[0], reverse=True)

    return prob_and_disease[0][0].item(), prob_and_disease[1][0].item(), prob_and_disease[0][1], prob_and_disease[1][1]

# @app.post("/predict")
# async def predict_disease(item: Item):
#     prob1, prob2, disease_name1, disease_name2 = predict(item.image_path, item.crop)
#
#     return {"disease1": disease_name1, "probability1": prob1, "disease2": disease_name2, "probability2": prob2}