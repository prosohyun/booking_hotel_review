#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
pd.set_option('display.max_row',500)
pd.set_option('display.max_columns',100)
pd.set_option('display.unicode.east_asian_width',True)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


# In[108]:


def getRecommendation(cosine_sim):
    simScores = list(enumerate(cosine_sim[-1])) #enumerate()열거해줌
    simScores = sorted(simScores, key=lambda x: x[1], reverse=True) #내림차순
    simScores = simScores[2:31]
    hotel_idx = [i[0] for i in simScores]
    RecHotellist = df_review_one_sentence.iloc[hotel_idx]
    return RecHotellist.hotel_name


# In[84]:


df_review_one_sentence = pd.read_csv('./hotel/onesentence_hotel_review_final.csv', index_col=0)
#print(df_review_one_sentence.head())
print(df_review_one_sentence.info())


# In[104]:


hotel_idx = df_review_one_sentence[df_review_one_sentence['hotel_name']=='제주 아름다운 리조트'].index[0] #인덱스 알아내는 방법


# In[105]:


Tfidf = TfidfVectorizer()
Tfidf_matrix = Tfidf.fit_transform(df_review_one_sentence['review_one_sentence'])
#print(Tfidf_matrix.shape)
#print(Tfidf_matrix)


# In[106]:


cosine_sim = linear_kernel(Tfidf_matrix[hotel_idx], Tfidf_matrix) #호텔1개에 대한 588개호텔의 수치
#print(cosine_sim.shape)


# In[109]:


print(getRecommendation(cosine_sim))


# In[ ]:




