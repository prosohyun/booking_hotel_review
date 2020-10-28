#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
pd.set_option('display.max_row',500)
pd.set_option('display.max_columns',100)
pd.set_option('display.unicode.east_asian_width',True)
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from gensim.models import Word2Vec
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import collections


# In[73]:


fontpath = './malgun.ttf'


# In[79]:


review_words = pd.read_csv('./hotel/onesentence_hotel_review_final.csv', index_col=0)
print(review_words.columns)
print(review_words['hotel_name'].unique())
#print(review_words.iloc[0]['review_one_sentence'])


# In[75]:


movie_idx = review_words[review_words['hotel_name']=='리조트 켄싱턴리조트 설악비치'].index[0]


# In[76]:


tokend_review_words = review_words['review_one_sentence'][movie_idx].split(' ')


# In[77]:


textdict = collections.Counter(tokend_review_words)


# In[78]:


wordcloud_img = WordCloud(background_color='white', max_words=2000, font_path=fontpath).generate_from_frequencies(textdict)
plt.figure(figsize=(8,8))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off') # x값 y값 안보여줌
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




