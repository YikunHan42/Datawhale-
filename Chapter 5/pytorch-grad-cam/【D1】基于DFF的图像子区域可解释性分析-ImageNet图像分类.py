#!/usr/bin/env python
# coding: utf-8

# # 基于DFF的图像子区域可解释性分析
# 
# 对单张图像，进行Deep Feature Factorization可解释性分析，展示Concept Discovery概念发现图。
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-9-19

# ## 参考阅读
# 
# 代码库 pytorch-grad-cam：https://github.com/jacobgil/pytorch-grad-cam
# 
# 博客 Deep Feature Factorizations for better model explainability：https://jacobgil.github.io/pytorch-gradcam-book/Deep%20Feature%20Factorizations.html
# 
# 论文 Deep Feature Factorization For Concept Discovery：https://arxiv.org/abs/1806.10206

# ## 导入工具包

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import requests

from PIL import Image
import numpy as np
import pandas as pd
import cv2
import json

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
from pytorch_grad_cam import GradCAM
from torchvision.models import resnet50

import torch


# ## 预处理函数

# In[2]:


def get_image_from_path(img_path):
    '''
    输入图像文件路径，输出 图像array、归一化图像array、预处理后的tensor
    '''

    img = np.array(Image.open(img_path))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor


# In[3]:


def create_labels(concept_scores, top_k=2):
    """ Create a list with the image-net category names of the top scoring categories"""

    df = pd.read_csv('imagenet_class_index.csv')
    labels = {}
    for idx, row in df.iterrows():
        labels[row['ID']] = row['class']
    
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]    
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{labels[category].split(',')[0]}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk


# ## 载入模型

# In[4]:


model = resnet50(pretrained=True).eval()


# ## 载入测试图像

# In[5]:


img_path = 'test_img/cat_dog.jpg'


# In[6]:


# Image.open(img_path)


# ## 预处理

# In[7]:


img, rgb_img_float, input_tensor = get_image_from_path(img_path)


# In[8]:


img.shape


# In[9]:


input_tensor.shape


# ## 初始化DFF算法

# In[10]:


classifier = model.fc


# In[11]:


dff = DeepFeatureFactorization(model=model, 
                               target_layer=model.layer4, 
                               computation_on_concepts=classifier)


# In[12]:


# concept个数（图块颜色个数）
n_components = 5

concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)


# In[13]:


concepts.shape


# ## 图像中每个像素对应的concept热力图

# In[14]:


# concept个数 x 高 x 宽
batch_explanations[0].shape


# In[15]:


plt.imshow(batch_explanations[0][4])
plt.show()


# ## concept与类别的关系

# In[16]:


concept_outputs.shape


# In[17]:


concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()    


# In[18]:


concept_outputs.shape


# ## 每个concept展示前top_k个类别

# In[19]:


# 每个概念展示几个类别
top_k = 2


# In[20]:


concept_label_strings = create_labels(concept_outputs, top_k=top_k)


# In[21]:


concept_label_strings


# ## 生成可视化效果

# In[22]:


from pytorch_grad_cam.utils.image import show_factorization_on_image
visualization = show_factorization_on_image(rgb_img_float, 
                                            batch_explanations[0],
                                            image_weight=0.3, # 原始图像透明度
                                            concept_labels=concept_label_strings)


# In[23]:


result = np.hstack((img, visualization))


# In[23]:


Image.fromarray(result)


# ## 封装函数

# In[ ]:


def dff_show(img_path='test_img/cat_dog.jpg', n_components=5, top_k=2, hstack=False):
    img, rgb_img_float, input_tensor = get_image_from_path(img_path)
    dff = DeepFeatureFactorization(model=model, 
                                   target_layer=model.layer4, 
                                   computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)
    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()
    concept_label_strings = create_labels(concept_outputs, top_k=top_k)
    visualization = show_factorization_on_image(rgb_img_float, 
                                                batch_explanations[0],
                                                image_weight=0.3, # 原始图像透明度
                                                concept_labels=concept_label_strings)
    if hstack:
        result = np.hstack((img, visualization))
    else:
        result = visualization
    display(Image.fromarray(result))


# In[ ]:


dff_show()


# In[ ]:


dff_show(hstack=True)


# In[ ]:


dff_show(img_path='test_img/box_tabby.png', hstack=True)


# In[ ]:


dff_show(img_path='test_img/puppies.jpg', hstack=True)


# In[ ]:


dff_show(img_path='test_img/bear.jpeg', hstack=True)


# In[ ]:


dff_show(img_path='test_img/bear.jpeg', n_components=10, top_k=1, hstack=True)


# In[ ]:


dff_show(img_path='test_img/giraffe_zebra.jpg', n_components=5, top_k=2, hstack=True)


# In[ ]:




