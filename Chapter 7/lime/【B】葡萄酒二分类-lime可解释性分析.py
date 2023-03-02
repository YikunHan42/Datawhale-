#!/usr/bin/env python
# coding: utf-8

# # 葡萄酒数据集二分类+LIME可解释性分析
# 
# 在葡萄酒质量二分类数据集上训练随机森林分类模型，对测试集样本预测结果，基于LIME进行可解释性分析。
# 
# 定量评估出某个样本、某个特征，对模型预测为某个类别的贡献影响。
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 2022-10-30

# ## 导入工具包

# In[1]:


import numpy as np
import pandas as pd

import lime
from lime import lime_tabular


# ## 载入数据集
# 
# 数据集链接：https://www.kaggle.com/datasets/piyushagni5/white-wine-quality
# 
# https://archive.ics.uci.edu/ml/datasets/wine+quality

# In[2]:


df = pd.read_csv('wine.csv')


# In[3]:


df.shape


# In[6]:


df


# ## 划分训练集和测试集

# In[4]:


from sklearn.model_selection import train_test_split

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


X_train.shape


# In[6]:


X_test.shape


# In[7]:


y_train.shape


# In[8]:


y_test.shape


# ## 训练模型

# In[9]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# ## 评估模型

# In[10]:


score = model.score(X_test, y_test)


# In[11]:


score


# ## 初始化LIME可解释性分析算法

# In[13]:


explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train), # 训练集特征，必须是 numpy 的 Array
    feature_names=X_train.columns, # 特征列名
    class_names=['bad', 'good'], # 预测类别名称
    mode='classification' # 分类模式
)


# ## 从测试集中选取一个样本，输入训练好的模型中预测，查看预测结果

# In[17]:


# idx = 1

idx = 3


# In[18]:


data_test = np.array(X_test.iloc[idx]).reshape(1, -1)
prediction = model.predict(data_test)[0]
y_true = np.array(y_test)[idx]
print('测试集中的 {} 号样本, 模型预测为 {}, 真实类别为 {}'.format(idx, prediction, y_true))


# ## 可解释性分析

# In[19]:


exp = explainer.explain_instance(
    data_row=X_test.iloc[idx], 
    predict_fn=model.predict_proba
)

exp.show_in_notebook(show_table=True)


# ## 参考资料
# 
# UCI心脏病二分类+可解释性分析：https://www.bilibili.com/video/BV1Wf4y1U7EL
# 
# https://towardsdatascience.com/lime-how-to-interpret-machine-learning-models-with-python-94b0e7e4432e

# In[ ]:




