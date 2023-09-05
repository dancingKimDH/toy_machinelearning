# conda install -c conda-forge fastapi uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
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
    # !float으로 변수 datatype 지정해 주기!
    
    try:
        hypertension = float(data['hypertension'])
        gender = float(data['gender'])
        liver_status = float(data['liver_status'])
        age = float(data['age'])
        weight = float(data['weight'])
        surgery_duration = float(data['surgery_duration'])


        import os
        if not os.path.exists('datasets/BreastCancerWisconsin_Regression.pkl'):
            raise HTTPException(status_code=500, detail='Model file not found')

        # pkl 파일 존재 확인 코드 필요

        result_predict = 0
        # 학습 모델 불러와 예측
        with open('datasets/BreastCancerWisconsin_Regression.pkl', 'rb') as regression_file:
            loaded_model = pickle.load(regression_file)
            input_labels = [[hypertension, gender, liver_status, age, weight, surgery_duration]] # 학습했던 설명변수 형식 맞게 적용
            result_predict = loaded_model.predict(input_labels)
            print('Predict radius_mean Result : {}'.format(result_predict))
            pass

        # 예측값 리턴
        result = {'radius_mean':result_predict[0]}
        return result

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f'Missing or invalid data field: {e}')
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f'Invalid data type: {e}')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error: {e}')