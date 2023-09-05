#!/usr/bin/env python
# coding: utf-8

# #### 목표변수와 설명변수
# - Subject: 환자
# - Goal: Features가 Target인 수술실패여부에 미치는 영향 분석
# - Target: '수술실패여부'
# - Features: '고혈압여부', '성별', '신부전여부', '연령', '체중', '수술시간'

# In[2]:


import pandas as pd


# In[3]:


df_ROS = pd.read_csv('../datasets/RecurrenceOfSurgery.csv')
df_ROS_select = df_ROS[['수술실패여부', '고혈압여부', '성별', '신부전여부', '연령', '체중', '수술시간']]
df_ROS_select[:2]


# In[4]:


df_ROS.info()


# In[ ]:





# In[5]:


df_ROS_select.isnull().sum()


# In[6]:


# 수술시간이 Null인 행 삭제
df_ROS_select = df_ROS_select.dropna(subset=['수술시간'], how='any', axis=0)


# In[7]:


df_ROS_select.isnull().sum()


# #### 정형화

# In[8]:


target = df_ROS_select['수술실패여부']
features = df_ROS_select[['연령', '체중', '수술시간']]
target.shape, features.shape


# In[9]:


from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 10)
features_train.shape, features_test.shape, target_train.shape, target_test.shape


# #### 모델학습

# In[10]:


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
model = DecisionTreeClassifier()
# Target이 범주형이므로 Classifier을 사용


# In[11]:


from sklearn.model_selection import GridSearchCV
hyper_params = {'min_samples_leaf' : range(2,5),
               'max_depth' : range(2,5),
               'min_samples_split' : range(2,5)}


# #### 평가 Score Default, 분류(Accuracy), 예측(R square)

# In[12]:


from sklearn.metrics import f1_score, make_scorer
scoring = make_scorer(f1_score)


# In[13]:


grid_search = GridSearchCV(model, param_grid = hyper_params, cv=3, verbose=1, scoring=scoring)
grid_search.fit(features, target)


# In[14]:


grid_search.best_estimator_


# In[20]:


grid_search.best_score_, grid_search.best_params_

# 전처리 전의 정확도(accuracy) : 0.028571428571428574
# 낮은 정확도, 모델이 예측을 잘 수행하지 못함


# In[16]:


best_model = grid_search.best_estimator_
best_model


# In[17]:


target_test_predict = best_model.predict(features_test)
target_test_predict


# In[18]:


from sklearn.metrics import classification_report


# In[19]:


print(classification_report(target_test, target_test_predict))

# 전처리 전의 값
#  precision  recall  f1-score   support
#0  0.94      1.00      0.97       516
#1  1.00      0.03      0.05        36


# ### 결론
# - 의학 통계에서는 실제 instances들 중 해당하는 것을 identify하는 정도를 측정하는 recall이 더 중요함
