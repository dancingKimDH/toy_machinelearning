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

# In[3]:


import pandas as pd


# In[4]:


df_ROS = pd.read_csv('../datasets/RecurrenceOfSurgery.csv')
df_ROS[:2]


# In[5]:


df_ROS['수술시간'].info()


# #### 전처리

# In[6]:


df_ROS_select = df_ROS[['수술실패여부', '고혈압여부', '성별', '신부전여부', '연령', '체중', '수술시간']]
df_ROS_select[:2]


# #### Scaling & Encoding & Concat

# ##### - OneHotEncoding

# In[7]:


# 범주형 데이터 확인 : '고혈압여부', '성별', '신부전여부'
df_ROS_select['고혈압여부'].value_counts(),df_ROS_select['성별'].value_counts(),df_ROS_select['신부전여부'].value_counts()


# In[8]:


from sklearn.preprocessing import OneHotEncoder


# In[9]:


# 범주형 설명변수 OneHotEncoding
oneHotEncoder = OneHotEncoder() # 인스턴스화
oneHotEncoder.fit(df_ROS_select[['고혈압여부', '성별', '신부전여부']])


# In[10]:


oneHotEncoder.categories_


# In[11]:


encoded_data = oneHotEncoder.transform(df_ROS_select[['고혈압여부', '성별', '신부전여부']]).toarray()
encoded_data.shape


# In[12]:


df_encoded_data = pd.DataFrame(data=encoded_data, columns=oneHotEncoder.get_feature_names_out(['고혈압여부', '성별', '신부전여부']))
df_encoded_data[:2]


# ##### - 병합(Concat)

# In[13]:


df_ROS_select= pd.concat([df_ROS_select.reset_index(drop=True), df_encoded_data.reset_index(drop=True)], axis=1)
df_ROS_select[:2]


# In[14]:


df_ROS_select.shape


# ##### - Scaling

# In[15]:


df_ROS_select.columns


# In[16]:


target = df_ROS_select['수술실패여부']
features = df_ROS_select.drop(columns=['수술실패여부', '고혈압여부', '성별', '신부전여부'])


# In[17]:


features.columns


# ##### - MinMaxScaler

# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


minMaxScaler = MinMaxScaler() #인스턴스화
features= minMaxScaler.fit_transform(features)
features.shape


# #### 모델학습, apply()함수를 사용하여 null값 채우기
# - null 값 채우기 : '수술시간'

# In[20]:


# null값 확인 -> 수술시간 null값 존재
df_ROS_select.isnull().sum()


# In[21]:


# null값 삭제 -> null값이 없는 데이터로 모델 학습시키기 위함
df_ROS_select_drop = df_ROS_select.dropna()
df_ROS_select_drop.isnull().sum()


# In[22]:


# null값이 없는 데이터로 모델학습 준비
# 1. target : '수술시간', feature: '성별'
target = df_ROS_select_drop[['수술시간']]
feature = df_ROS_select_drop[['성별']]
target.shape, feature.shape


# In[23]:


# 2. null값이 없는 데이터와 실제값을 사용하여 회귀모델 훈련
from sklearn.linear_model import LinearRegression
model = LinearRegression() # 인스턴스화(초기화)
model.fit(feature, target) # 모델 훈련


# In[24]:


# 모델 예측 확인해보기 (type : numpy의 array)
model.predict(feature)


# In[25]:


# apply()
import numpy as np
def convert_notnull(row) :
    if pd.isnull(row) : # 변수 row의 값이 null이라면
        feature = df_ROS_select[['성별']]
        result = model.predict(feature)
        return result
    else :
        return row  # null이 아니면 원래 데이터 값 반환


# In[26]:


df_ROS_select['수술시간'] = df_ROS_select['수술시간'].apply(convert_notnull)
df_ROS_select['수술시간']


# In[27]:


# apply() 적용 후 null값 확인
df_ROS_select['수술시간'].isnull().sum()


# In[28]:


df_ROS_select.isnull().sum()


# #### Imbalanced Data Sampling
# - under sampling : Tomek's Link

# In[29]:


from imblearn.under_sampling import TomekLinks


# In[30]:


from sklearn.datasets import make_classification


# In[31]:


features, target = make_classification(n_classes=2, class_sep=2,
                    weights=[0.4, 0.6], n_informative=3, n_redundant=1, flip_y=0,
                    n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)


# In[32]:


features.shape, target.shape


# In[33]:


from collections import Counter


# In[34]:


Counter(target)


# In[35]:


tomekLinks = TomekLinks() #인스턴스화
features_resample, target_resample = tomekLinks.fit_resample(features, target) #교육


# In[36]:


features_resample.shape, target_resample.shape


# In[37]:


Counter(target_resample)


# #### 서비스 배포

# In[43]:


# 데이터를 저장하고 불러올 때 매우 유용한 라이브러리
# 클래스 자체를 통째로 파일로저장했다가 그것을 그대로 불러올 수 있음
import pickle


# In[39]:


# scaling model
with open('../../datasets/RecurrenceOfSurgery_scaling.pkl','wb') as scaling_file :
    pickle.dump(obj=model, file=scaling_file)
    pass


# In[ ]:


# encoding model
with open('','wb') as encoding_file :
    

