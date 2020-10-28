#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('display.max_row',500)
pd.set_option('display.max_columns',100)
pd.set_option('display.unicode.east_asian_width',True)
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


# In[2]:


from matplotlib import font_manager, rc # 한글폰트사용
import matplotlib
font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
matplotlib.rcParams['axes.unicode_minus']=False
rc('font', family=font_name)


# In[3]:


embedding_model = Word2Vec.load('./model/word2VecModel_hotel.model')
print(embedding_model.wv.vocab.keys())
print(len(embedding_model.wv.vocab.keys()))


# In[24]:


key_word = '산책길'
sim_word = embedding_model.wv.most_similar(key_word, topn=30)
print(sim_word)


# In[25]:


print(sim_word[0][0])
print(embedding_model[sim_word[0][0]]) # 100차원짜리 벡터출력


# In[26]:


tokens = []
labels = []
for i in sim_word:
    labels.append(i[0])
    tokens.append(embedding_model[i[0]])


# In[27]:


print(tokens[0])
print(len(tokens[0]))
print(labels)


# In[28]:


tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)


# In[29]:


df_tokens = pd.DataFrame(tokens)
print(df_tokens.head())


# In[30]:


new_values = tsne_model.fit_transform(df_tokens)
print(new_values[0])
print(type(new_values))


# In[31]:


df_xy = pd.DataFrame({'words':labels, 'x':new_values[:,0], 'y':new_values[:,1]})
print(df_xy.head())


# In[32]:


df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)
print(df_xy.tail())


# In[33]:


plt.figure(figsize=(8,8))
plt.scatter(0, 0, s=500, marker='*')
for i in range(len(df_xy.x)):
    a = df_xy.loc[{i, 30},:]
    plt.plot(a.x, a.y, '-D', linewidth=2)
    plt.scatter(df_xy.x[i], df_xy.y[i])
    plt.annotate(df_xy.words[i], xy=(df_xy.x[i], df_xy.y[i]), xytext=(5,2), textcoords='offset points', ha='right', va='bottom' )
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




