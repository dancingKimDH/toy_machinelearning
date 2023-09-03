# 퀘스트
- 업무분장(전처리, 모델학습).
- RecurrenceOfSurgery.csv 사용
- 목표변수 : 범주형, 설명변수 : 최소 6개 
- 서비스 대상과 목표 설명, 변수 선택 이유

## 📃 Authors
- 김혜인 
- 김동현 

## 🔬 분석 목적
- 서비스 대상 : 환자
- 목적 : '수술실패여부'에 영향을 줄 수 있는 요인들을 선정해 관계 확인 후 환자에게 설명을 해드리기 위함
- 목적변수(Target) : '수술실패여부'
- 설명변수(Features) : '고혈압여부', '성별', '신부전여부', '연령', '체중', '수술시간'

## 🌟 변수 선택 이유
|변수|변수 설명|변수타입|변수 선택 이유|
|---|---------|--------|----------------|
|수술실패여부|환자의 수술 실패 여부|범주형|환자에게 수술 실패에 영향을 미칠 수 있는 여러 요인들을 설명하고자 함|
|고혈압여부|환자의 고혈압 여부|범주형|고혈압이 있을 경우 수술실패에 영향을 미치는지 확인하고자 함|
|성별|환자의 성별|범주형|성별이 수술실패에 영향을 미치는지 확인하고자 함|
|신부전여부|환자의 신부전 여부|범주형|신부전이 있을 경우 수술실패에 영향을 미치는지 확인하고자 함|
|연령|환자의 연령|연속형|환자의 연령에 따라 수술실패여부가 달라지는지 확인하고자 함 |
|체중|환자의 체중|연속형|환자의 체주에 따라 수술실패여부가 달라지는지 확인하고자 함|
|수술시간|환자의 수술시간|연속형|수술시간이 오래 걸릴수록 수술실패여부가 달라지는지 확인하고자 함|

## 🔎 분석 결과
- 전처리 전의 값

|  |precision|recall|f1-score|support|
|--|---------|------|--------|-------|
|0 |0.94|1.00|0.97|516|
|1 |1.00|0.03|0.05|36|

- 전처리 후의 값

|  |precision|recall|f1-score|support|
|--|---------|------|--------|-------|
|0 |||||
|1 |||||

## 느낀점
- 김동현 :
- 김혜인 :
