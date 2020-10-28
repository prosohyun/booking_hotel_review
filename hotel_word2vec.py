#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
pd.set_option('display.max_row',500)
pd.set_option('display.max_columns',100)
pd.set_option('display.unicode.east_asian_width',True)
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from gensim.models import Word2Vec


# In[23]:


review_words = pd.read_csv('./hotel/cleaned_hotel_review_final.csv', index_col=0)
print(review_words.columns)
print(review_words.iloc[0]['cleaned_review'])


# In[24]:


print(review_words.head())
print(review_words.info())


# In[20]:


clean_token_review = list(review_words['cleaned_review'])
cleaned_sentences = []
for sentence in clean_token_review:
    token = sentence.split(' ')
    cleaned_sentences.append(token)
print(cleaned_sentences[0])
print(type(cleaned_sentences))


# In[21]:


embedding_model = Word2Vec(cleaned_sentences, size=100, window=4, min_count=20, workers=4, iter=100, sg=1)
embedding_model.save('./model/word2VecModel_hotel.model')


# In[22]:


print(embedding_model.wv.vocab.keys())
print(len(embedding_model.wv.vocab.keys()))


# In[ ]:





# In[ ]:





# In[ ]:




