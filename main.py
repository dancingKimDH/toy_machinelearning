# conda install -c conda-forge fastapi uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# No 'Access-Control-Allow-Origin'
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 접근 가능한 도메인만 허용하는 것이 좋습니다.
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

import pickle

# /api_v1/mlmodelwithregression with dict params
# method : post
@app.post('/api_v1/mlmodelwithregression') 
def mlmodelwithregression(data:dict) : 
    print('data with dict {}'.format(data))
    
    # data dict to 변수 할당
    hypertension = float(data['hypertension'])
    gender = float(data['gender'])
    liver_status = float(data['liver_status'])
    age = float(data['age'])
    weight = float(data['weight'])
    surgery_duration = float(data['surgery_duration'])

    # pkl 파일 존재 확인 코드 필요
    import numpy as np

    # OneHotEncoding.pkl 불러오기
    with open('datasets/RecurrenceOfSurgery_encoding.pkl', 'rb') as encoding_file:
        loaded_model = pickle.load(encoding_file)
        input_encoding_labels = [[hypertension, gender, liver_status]] # 학습했던 설명변수 형식 맞게 적용
        result_predict1 = loaded_model.transform(input_encoding_labels)
        print('Predict Encoding Result : {}'.format(result_predict1))
        pass

    result_predict1_dense = result_predict1.toarray()
    # scaling.pkl 불러오기
    with open('datasets/RecurrenceOfSurgery_scaling.pkl', 'rb') as scaling_file:
        loaded_model = pickle.load(scaling_file)
        input_scaler_labels = [[age, weight, surgery_duration]+list(result_predict1_dense[0])] # 학습했던 설명변수 형식 맞게 적용
        result_predict2 = loaded_model.transform(input_scaler_labels)
        print('Predict Scaler Result : {}'.format(result_predict2))
        pass

    result_predict = 0
    # best model 불러와 예측
    with open('datasets/RecurrenceOfSurgery_best_model.pkl', 'rb') as bestmodel_file:
        loaded_model = pickle.load(bestmodel_file)
        input_labels = np.array(result_predict2) # 학습했던 설명변수 형식 맞게 적용
        result_predict = loaded_model.predict(input_labels)
        print('Predict Final Result : {}'.format(result_predict))
        pass

    # 예측값 리턴
    result = dict({'result_mean':result_predict[0]})
    return result