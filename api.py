#도커에 export MLFLOW_TRACKING_URI을 넣어주고 aws 엑세스 키와 시크릿 키를 넣어줌

import io
from fastapi import APIRouter, File

import mlflow

#이미지 처리
from PIL import Image

import cv2
import numpy as np

## 백엔드가 기본 모델 하나를 가질 수 있도록 설정
## workspace에 xml파일을 추가
default_model_uri = "~~~~~~~~~~~~tensorflow-model"
haarcascade_frontalface_default_path ="/workspace/haarcascade_frontalface_default.xml"

router = APIRouter()

model = mlflow.keras.load_model(default_model_uri)
face_model = cv2.CascadeClassifier(haarcascade_frontalface_default_path) #위에서 xml파일을 불러오며 face_model을 불러옴

@router.get("/hello/world")
def hello_world():
    return "hello minseok"
## url형식으로 인자를 받게 됨 
## swagger은 하나의 문서
@router.post("/model/{run_id}/{model_name}") 

# 모델을 선택하는 모델, run_id와 model_name을 받고 model_uri에 run_id와 model_name이 들어감
# 이를 model에 넣어줌
def select_model(
    run_id: str,
    model_name: str,
):
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.keras.load_model(model_uri)
    
    return "select model successful!" 

#이미지를 요청하고 이미지에 마스크를 쓰고있는 사람의 위치, 쓰지않은 사람의 위치를 결정(추론하는 api)
#비동기 예측(async)
# bytes이미지를 pillow로 바꿔줌
# rgb로 받아옴
@router.post("/predict")
async def predict(file: bytes=File(...)):
    input_images = Image.open(io.BytesIO(file)).convert("RGB")
    img = cv2.cvtColor(np.array(input_images), cv2.IMREAD_GRAYSCALE)  ### 받아온 이미지를 처리(이미지를 grayscale로 바꿔줌) 
    faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4) # 얼굴이 어딨는지 찾기

    mask_label = {0:'True',1:'False'}  
    results = {"is_mask": [], "bbox": []} #결과를 두 개로 반환
#하나의 이미지에 마스크가 몇개가 들어 있는지 그리고 그 마스크의 위치가 어딘지
    if len(faces)>=1:
        label = [0 for i in range(len(faces))]
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            crop = new_img[y:y+h,x:x+w]
            crop = cv2.resize(crop,(128,128))
            crop = np.reshape(crop,[1,128,128,3])/255.0
            mask_result = model.predict(crop)
            print(x,y,w,h)
            print(mask_label[mask_result.argmax()])


            results["is_mask"].append(mask_label[mask_result.argmax()])
            results["bbox"].append([int(x), int(y), int(x+w), int(y+h)])  # numpy에서 읽어왔으므로 int형으로 변환, 
						                                                  # 좌표 값이 좌측 위에 (0, 0)이 있고 우하향하며 커지게 된다.
    
    return results

