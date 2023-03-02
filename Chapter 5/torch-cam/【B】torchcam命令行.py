#!/usr/bin/env python
# coding: utf-8

# # torchcam可解释性分析可视化-命令行
# 
# 通过命令行方式使用torchcam算法库，对图像进行各种基于CAM的可解释性分析。
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-8-19

# ## 导入工具包

# In[1]:


import os

import pandas as pd

from PIL import Image


# ## 命令行基本用法

# In[4]:


get_ipython().system('python torch-cam/scripts/cam_example.py --help')


# ## ImageNet预训练图像分类模型

# In[2]:


# ImageNet1000类别名称与ID号
df = pd.read_csv('imagenet_class_index.csv')


# In[3]:


df


# 一些类别的名称与ID
# 
# 网球 852
# 
# basketball 430
# 
# cowboy_boot 514
# 
# 边牧犬 232
# 
# 牧羊犬 231
# 
# 虎斑猫 282、281
# 
# 网球 852
# 

# ## 图中只有一个类别

# In[5]:


# 类别-边牧犬
get_ipython().system('python torch-cam/scripts/cam_example.py         --img test_img/border-collie.jpg         --savefig output/B1_border_collie.jpg         --arch resnet18         --class-idx 232         --rows 2')


# In[6]:


Image.open('output/B1_border_collie.jpg')


# ## 图中有多个类别

# In[7]:


# 类别-虎斑猫
get_ipython().system('python torch-cam/scripts/cam_example.py         --img test_img/cat_dog.jpg         --savefig output/B2_cat_dog.jpg         --arch resnet18         --class-idx 282         --rows 2')


# In[8]:


Image.open('output/B2_cat_dog.jpg')


# In[10]:


# 类别-边牧犬
get_ipython().system('python torch-cam/scripts/cam_example.py         --img test_img/cat_dog.jpg         --savefig output/B3_cat_dog.jpg         --arch resnet18         --class-idx 232         --rows 2')


# In[12]:


Image.open('output/B3_cat_dog.jpg')


# In[ ]:




