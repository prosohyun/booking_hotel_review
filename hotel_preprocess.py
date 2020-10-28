#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from konlpy.tag import Okt


# In[2]:


df = pd.read_csv('./hotel/review_all.csv', index_col=0) #데이터불러오기
print(df.info())


# In[3]:


hotel_review = df['hotel_review'] # concat 및 컬럼삭제
hotel_tourlist = df['hotel_tourlist']
df['hotel_concat'] = pd.concat([hotel_review, hotel_tourlist], ignore_index=True)
df.drop(columns=['hotel_review','hotel_tourlist'], inplace=True)
print(df.info())


# In[4]:


df.drop_duplicates(inplace=True) # 중복제거
print(df.info())


# In[5]:


df = df.rename({'hotel_concat':'hotel_review'}, axis=1) # hotel_review로 이름변경
print(df.info())


# In[6]:


df.dropna(subset=['hotel_review'], how='any', inplace=True)
print(df.info())


# In[7]:


okt = Okt()


# In[8]:


stopwords = pd.read_csv('../prj_news_category_classfication/datasets/stopwords_hotel.csv', index_col=0)
#print(stopwords)


# In[9]:


cleaned_sentences = []
tokened_sentences = []
for sentence in df['hotel_review']:
    sentence = re.sub('[^가-힣]',' ',sentence)
    token = okt.pos(sentence, norm=True, stem=True)
    df_token = pd.DataFrame(token,columns=['word','class'])
    cleaned_df_token = df_token[(df_token['class']=='Noun')|(df_token['class']=='Verb')|(df_token['class']=='Adjective')]
    words = []
    for word in cleaned_df_token['word']:
        if len(word) > 1:
            if word not in (list(stopwords['stopword'])):
                words.append(word)
            else: print(word)
        else: print(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)
    tokened_sentences.append(words)
df['tokened_review'] = tokened_sentences
df['cleaned_review'] = cleaned_sentences


# In[10]:


df.to_csv('./hotel/cleaned_hotel_review_final.csv') # 숫자변경
print(df.info())


# In[11]:


df = pd.read_csv('./hotel/cleaned_hotel_review_final.csv', index_col=0) # 숫자변경
df.dropna(subset=['cleaned_review'], how='any', inplace=True) # 결측치제거
df.reset_index(drop=True, inplace=True)
print(df.info())
print(df['hotel_name'].unique())


# In[12]:


df.to_csv('./hotel/cleaned_hotel_review_final.csv') # 숫자변경


# In[13]:


review_one_sentences = []
hotel_names = []
for hotel_name in df['hotel_name'].unique():
    temp = df[df['hotel_name']==hotel_name]['cleaned_review']
    review_one_sentence = ' '.join(list(temp))
    review_one_sentences.append(review_one_sentence)
    hotel_names.append(hotel_name)
df_review_one_sentence = pd.DataFrame({'hotel_name':hotel_names, 'review_one_sentence':review_one_sentences})
df_review_one_sentence.to_csv('./hotel/onesentence_hotel_review_final.csv') # 숫자변경
print(df_review_one_sentence.head())
print(df_review_one_sentence.info())

