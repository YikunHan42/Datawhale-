#!/usr/bin/env python
# coding: utf-8

# # Tensorflow-预训练ResNet50局部遮挡可解释性分析
# 
# 将输入图像局部遮挡，对ResNet50图像分类模型的预测结果进行可解释性分析。
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-10-24

# ## 导入工具包

# In[1]:


import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import shap


# ## 导入预训练模型

# In[2]:


model = ResNet50(weights='imagenet')


# ## 导入数据集

# In[3]:


X, y = shap.datasets.imagenet50()


# ## 构建模型预测函数

# In[5]:


def f(x):
    tmp = x.copy()
    preprocess_input(tmp)
    return model(tmp)


# ## 构建局部遮挡函数

# In[6]:


masker = shap.maskers.Image("inpaint_telea", X[0].shape)


# ## 输出类别名称

# In[7]:


url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]


# In[9]:


# class_names


# ## 创建Explainer

# In[10]:


explainer = shap.Explainer(f, masker, output_names=class_names)


# ## 计算shap值

# In[11]:


shap_values = explainer(X[1:3], max_evals=100, batch_size=50, outputs=shap.Explanation.argsort.flip[:4]) 


# ## 可视化

# In[6]:


shap.image_plot(shap_values)


# > 原图可视化如果有误，不用担心，不影响后面几个图的shap可视化效果。

# ## 更加细粒度的shap计算和可视化

# In[13]:


masker_blur = shap.maskers.Image("blur(128,128)", X[0].shape)

explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names)

shap_values_fine = explainer_blur(X[1:3], max_evals=5000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4]) 

shap.image_plot(shap_values_fine)


# In[7]:


shap.image_plot(shap_values_fine)


# > 原图可视化如果有误，不用担心，不影响后面几个图的shap可视化效果。

# In[ ]:





# In[ ]:




