#!/usr/bin/env python
# coding: utf-8

# # GradientShap可解释性分析
# 
# 使用GradientShap，在输入图像上，对指定类别进行可解释性分析。
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-8-19

# ## 导入工具包

# In[2]:


import os
import json
import numpy as np
import pandas as pd
np.random.seed(0)

from PIL import Image

import torch
torch.manual_seed(0)
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import transforms

# from captum.attr import IntegratedGradients
from captum.attr import GradientShap
# from captum.attr import Occlusion
# from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
get_ipython().run_line_magic('matplotlib', 'inline')

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 载入自己训练好的模型

# In[3]:


model = torch.load('checkpoints/fruit30_pytorch_20220814.pth')
model = model.eval().to(device)


# ## 载入类别

# In[4]:


idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()


# In[5]:


idx_to_labels


# ## 图像预处理

# In[6]:


from torchvision import transforms

# 缩放、裁剪、转 Tensor、归一化
transform_A = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),          
    transforms.ToTensor()         
])

transform_B = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# ## 载入测试图像

# In[15]:


img_path = 'test_img/test_lemon.jpg'


# In[16]:


img_pil = Image.open(img_path)


# In[17]:


img_pil


# ## 预处理

# In[18]:


# 缩放、裁剪
rc_img = transform_A(img_pil)

# 调整数据维度
rc_img_norm = np.transpose(rc_img.squeeze().cpu().detach().numpy(), (1,2,0))

# 色彩归一化
input_tensor = transform_B(rc_img).unsqueeze(0).to(device)


# ## 前向预测

# In[19]:


pred_logits = model(input_tensor)
pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算

pred_conf, pred_id = torch.topk(pred_softmax, 1)
pred_conf = pred_conf.detach().cpu().numpy().squeeze().item()
pred_id = pred_id.detach().cpu().numpy().squeeze().item()
pred_label = idx_to_labels[pred_id]

print('预测类别的ID {} 名称 {} 置信度 {:.2f}'.format(pred_id, pred_label, pred_conf))


# ## GradientShap可解释性分析

# GradientShap 是一种线性的模型可解释性分析方法，使用多张参考图像（在本例中为2张）解释模型预测结果。参考图像通过给定的 baseline 分布随机生成。计算`每个像素`分别采用`原始输入图像像素值`和`baseline图像像素值`的梯度期望。

# In[20]:


gradient_shap = GradientShap(model)

# 设置 baseline distribution
rand_img_dist = torch.cat([input_tensor * 0, input_tensor * 1])


# In[21]:


# 获得输入图像每个像素的 GradientShap 值
attributions_gs = gradient_shap.attribute(input_tensor,
                                          n_samples=50,
                                          stdevs=0.0001,
                                          baselines=rand_img_dist,
                                          target=pred_id)

# 转为 224 x 224 x 3的数据维度
attributions_gs_norm = np.transpose(attributions_gs.detach().cpu().squeeze().numpy(), (1,2,0))


# In[22]:


# 设置配色方案
default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

viz.visualize_image_attr_multiple(attributions_gs_norm,
                                  rc_img_norm,
                                  ["original_image", "heat_map"],
                                  ["all", "absolute_value"],
                                  cmap=default_cmap,
                                  show_colorbar=True)
plt.show()


# In[ ]:




