#!/usr/bin/env python
# coding: utf-8

# ## Quest
# - 업무분장(전처리, 모델학습).
# - RecurrenceOfSurgery.csv 사용
# - 목표변수 : 범주형, 설명변수 : 최소 6개 
# - 서비스 대상과 목표 설명, 변수 선택 이유

# ### 변수설정
# 0. 서비스 대상: 환자
#     * 수술실패에 영향을 미치는 요인들이 무엇이 있을까?
# 1. target: 수술실패여부
# 2. features 
#     * 범주형: 고혈압여부
#     * 범주형: 성별
#     * 범주형: 신부전여부
#     * 연속형: 연령
#     * 연속형: 체중
#     * 연속형: 수술시간

import pandas as pd

df_ROS = pd.read_csv('../datasets/RecurrenceOfSurgery.csv')
df_ROS_select = df_ROS[['수술실패여부', '고혈압여부', '성별', '신부전여부', '연령', '체중', '수술시간']]
df_ROS_select[:2]

df_ROS.info()

# #### 전처리

df_ROS_select = df_ROS[['수술실패여부', '고혈압여부', '성별', '신부전여부', '연령', '체중', '수술시간']]
df_ROS_select[:2]


# #### Scaling & Encoding & Concat
# ##### - OneHotEncoding
# 범주형 데이터 확인 : '고혈압여부', '성별', '신부전여부'
df_ROS_select['고혈압여부'].value_counts(),df_ROS_select['성별'].value_counts(),df_ROS_select['신부전여부'].value_counts()

from sklearn.preprocessing import OneHotEncoder

# 범주형 설명변수 OneHotEncoding
oneHotEncoder = OneHotEncoder() # 인스턴스화
oneHotEncoder.fit(df_ROS_select[['고혈압여부', '성별', '신부전여부']])

oneHotEncoder.categories_

encoded_data = oneHotEncoder.transform(df_ROS_select[['고혈압여부', '성별', '신부전여부']]).toarray()
encoded_data.shape

df_encoded_data = pd.DataFrame(data=encoded_data, columns=oneHotEncoder.get_feature_names_out(['고혈압여부', '성별', '신부전여부']))
df_encoded_data[:2]


# ##### - 병합(Concat)
df_ROS_select= pd.concat([df_ROS_select.reset_index(drop=True), df_encoded_data.reset_index(drop=True)], axis=1)
df_ROS_select[:2]

df_ROS_select.shape

# ##### - Scaling

df_ROS_select.columns

target = df_ROS_select['수술실패여부']
features = df_ROS_select.drop(columns=['수술실패여부', '고혈압여부', '성별', '신부전여부'])

features.columns


# ##### - MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

minMaxScaler = MinMaxScaler() #인스턴스화
features= minMaxScaler.fit_transform(features)
features.shape


# #### 모델학습, apply()함수를 사용하여 null값 채우기
# - null 값 채우기 : '수술시간'
# null값 확인 -> 수술시간 null값 존재
df_ROS_select.isnull().sum()

# null값 삭제 -> null값이 없는 데이터로 모델 학습시키기 위함
df_ROS_select_drop = df_ROS_select.dropna()
df_ROS_select_drop.isnull().sum()

# null값이 없는 데이터로 모델학습 준비
# 1. target : '수술시간', feature: '성별'
target = df_ROS_select_drop[['수술시간']]
feature = df_ROS_select_drop[['성별']]
target.shape, feature.shape

# 2. null값이 없는 데이터와 실제값을 사용하여 회귀모델 훈련
from sklearn.linear_model import LinearRegression
model = LinearRegression() # 인스턴스화(초기화)
model.fit(feature, target) # 모델 훈련

# 모델 예측 확인해보기 (type : numpy의 array)
model.predict(feature)

# apply()
import numpy as np
def convert_notnull(row) :
    if np.isnan(row['수술시간'])  : # 변수 row의 값이 null이라면
        feature = df_ROS_select[['성별']]
        result = model.predict(feature)
        return result[0]
    else :
        return row['수술시간']  # null이 아니면 원래 데이터 값 반환

df_ROS_select['수술시간'] = df_ROS_select.apply(convert_notnull, axis=1)
df_ROS_select['수술시간']

# apply() 적용 후 null값 확인
df_ROS_select['수술시간'].isnull().sum()

df_ROS_select.isnull().sum()

# #### Imbalanced Data Sampling
# - under sampling : Tomek's Link
from imblearn.under_sampling import TomekLinks
from sklearn.datasets import make_classification

features, target = make_classification(n_classes=2, class_sep=2,
                    weights=[0.4, 0.6], n_informative=3, n_redundant=1, flip_y=0,
                    n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

features.shape, target.shape

from collections import Counter

Counter(target)

tomekLinks = TomekLinks() #인스턴스화
features_resample, target_resample = tomekLinks.fit_resample(features, target) #교육

features_resample.shape, target_resample.shape

Counter(target_resample)

# #### 정형화
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 10)
features_train.shape, features_test.shape, target_train.shape, target_test.shape


# #### 모델학습
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
model = DecisionTreeClassifier()
# Target이 범주형이므로 Classifier을 사용

from sklearn.model_selection import GridSearchCV
hyper_params = {'min_samples_leaf' : range(2,5),
               'max_depth' : range(2,5),
               'min_samples_split' : range(2,5)}


# #### 평가 Score Default, 분류(Accuracy), 예측(R square)
from sklearn.metrics import f1_score, make_scorer
scoring = make_scorer(f1_score)

grid_search = GridSearchCV(model, param_grid = hyper_params, cv=3, verbose=1, scoring=scoring)
grid_search.fit(features, target)

grid_search.best_estimator_

grid_search.best_score_, grid_search.best_params_
# 전처리 전의 정확도(accuracy) : 0.028571428571428574
# 낮은 정확도, 모델이 예측을 잘 수행하지 못함

best_model = grid_search.best_estimator_
best_model

target_test_predict = best_model.predict(features_test)
target_test_predict

from sklearn.metrics import classification_report

print(classification_report(target_test, target_test_predict))