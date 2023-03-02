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

# In[10]:


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

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('device', device)


# ## 预处理函数

# In[11]:


from torchvision import transforms

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# In[12]:


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


# In[77]:


def create_labels(concept_scores, top_k=2):
    """ Create a list with the image-net category names of the top scoring categories"""
    
    labels = {
        0:'Hami Melon',
        1:'Cherry Tomatoes',
        2:'Shanzhu',
        3:'Bayberry',
        4:'Grapefruit',
        5:'Lemon',
        6:'Longan',
        7:'Pears',
        8:'Coconut',
        9:'Durian',
        10:'Dragon Fruit',
        11:'Kiwi',
        12:'Pomegranate',
        13:'Sugar orange',
        14:'Carrots',
        15:'Navel orange',
        16:'Mango',
        17:'Balsam pear',
        18:'Apple Red',
        19:'Apple Green',
        20:'Strawberries',
        21:'Litchi',
        22:'Pineapple',
        23:'Grape White',
        24:'Grape Red',
        25:'Watermelon',
        26:'Tomato',
        27:'Cherts',
        28:'Banana',
        29:'Cucumber'
    }
    
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

# In[78]:


model = torch.load('checkpoint/fruit30_pytorch_20220814.pth')
model = model.eval().to(device)


# ## 载入测试图像

# In[79]:


img_path = 'test_img/test_fruits.jpg'


# In[80]:


img_pil = Image.open(img_path)


# In[81]:


# img_pil


# In[82]:


input_tensor = test_transform(img_pil).unsqueeze(0).to(device)


# In[83]:


input_tensor.shape


# ## 预处理

# In[84]:


img, rgb_img_float, input_tensor = get_image_from_path(img_path)


# In[85]:


img.shape


# In[86]:


input_tensor.shape


# ## 初始化DFF算法

# In[87]:


classifier = model.fc


# In[88]:


dff = DeepFeatureFactorization(model=model, 
                               target_layer=model.layer4, 
                               computation_on_concepts=classifier)


# In[89]:


# concept个数（图块颜色个数）
n_components = 5

concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)


# In[90]:


concepts.shape


# ## 图像中每个像素对应的concept热力图

# In[91]:


# concept个数 x 高 x 宽
batch_explanations[0].shape


# In[92]:


plt.imshow(batch_explanations[0][2])
plt.show()


# ## concept与类别的关系

# In[93]:


concept_outputs.shape


# In[94]:


concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()    


# In[95]:


concept_outputs.shape


# ## 每个concept展示前top_k个类别

# In[96]:


# 每个概念展示几个类别
top_k = 2


# In[97]:


concept_label_strings = create_labels(concept_outputs, top_k=top_k)


# In[98]:


concept_label_strings


# ## 生成可视化效果

# In[99]:


from pytorch_grad_cam.utils.image import show_factorization_on_image
visualization = show_factorization_on_image(rgb_img_float, 
                                            batch_explanations[0],
                                            image_weight=0.3, # 原始图像透明度
                                            concept_labels=concept_label_strings)


# In[100]:


result = np.hstack((img, visualization))


# In[101]:


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


dff_show(img_path='test_img/test_草莓.jpg', hstack=True)


# In[ ]:


dff_show(img_path='test_img/test_火龙果.jpg', hstack=True)


# In[ ]:


dff_show(img_path='test_img/test_石榴.jpg', hstack=True)


# In[ ]:


dff_show(img_path='test_img/test_bananan.jpg', hstack=True)


# In[ ]:


dff_show(img_path='test_img/test_kiwi.jpg', hstack=True)


# In[ ]:





# In[ ]:





# In[ ]:




