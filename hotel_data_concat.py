#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[20]:


df = pd.read_csv('./hotel/cleaned_hotel_review_final.csv', index_col=0)


# In[17]:


for i in range(1,39): # 숫자바꾸기
    data = pd.read_csv('./hotel/cleaned_hotel_review_{0}.csv'.format(i), index_col=0)
    df = pd.concat([df, data], ignore_index=True)


# In[21]:


df.head()
df.info()


# In[22]:


df.dropna(subset=['cleaned_review'], how='any', inplace=True) #결측치제거
df.drop(columns='tokened_review', inplace=True) #토큰리뷰삭제
print(df.info())


# In[23]:


df.drop_duplicates(inplace=True) #중복제거
df.reset_index(drop=True, inplace=True)
print(df.info())


# In[24]:


df.to_csv('./hotel/cleaned_hotel_review_final.csv')


# In[32]:


df_one = pd.read_csv('./hotel/cleaned_hotel_review_final.csv', index_col=0)
df.drop_duplicates(inplace=True) #중복제거
df.reset_index(drop=True, inplace=True)
print(df.info())


# In[33]:


df.to_csv('./hotel/cleaned_hotel_review_final.csv')


# In[10]:


df_one = pd.read_csv('./hotel/onesentence_hotel_review_all.csv', index_col=0)
for i in range(1,39): # 숫자바꾸기
    data = pd.read_csv('./hotel/onesentence_hotel_review_{0}.csv'.format(i), index_col=0)
    print(i, data.info())
    df_one = pd.concat([df_one, data], ignore_index=True)
print(df_one.info())


# In[27]:


df_one.to_csv('./hotel/onesentence_hotel_review_final.csv')


# In[34]:


df_one = pd.read_csv('./hotel/onesentence_hotel_review_final.csv', index_col=0)
print(df.info())


# In[35]:


df_one.drop_duplicates(inplace=True) #중복제거
df_one.reset_index(drop=True, inplace=True)
print(df.info())


# In[36]:


df_one.to_csv('./hotel/onesentence_hotel_review_final.csv')


# In[ ]:





# In[ ]:





# In[15]:


# df_too = pd.read_csv('./hotel/review_1.csv', index_col=0)
# for i in range(2,31): # 숫자바꾸기
#     data = pd.read_csv('./hotel/review_{0}.csv'.format(i), index_col=0)
#     df_too = pd.concat([df_too, data], ignore_index=True)
# df_too.to_csv('./hotel/review_all.csv')


# In[ ]:





# In[ ]:




