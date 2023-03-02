#!/usr/bin/env python
# coding: utf-8

# # Grad-CAM热力图可解释性分析
# 
# 对单张图像，进行Grad-CAM可解释性分析。
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-9-21

# ## 导入工具包

# In[2]:


from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet50

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 载入ImageNet预训练图像分类模型

# In[3]:


model = resnet50(pretrained=True).eval().to(device)


# ## 图像预处理

# In[4]:


from torchvision import transforms

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(512),
                                     # transforms.CenterCrop(512),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# ## 载入测试图像

# In[4]:


img_path = 'test_img/cat_dog.jpg'


# In[5]:


img_pil = Image.open(img_path)


# In[6]:


# img_pil


# In[7]:


input_tensor = test_transform(img_pil).unsqueeze(0).to(device) # 预处理


# In[8]:


input_tensor.shape


# input_tensor 可以在 batch 维度有多张图片

# ## 指定分析的类别

# In[9]:


# 如果 targets 为 None，则默认为最高置信度类别
targets = [ClassifierOutputTarget(232)]


# 281 虎斑猫
# 
# 232 边牧犬

# ## 分析模型结构，确定待分析的层

# In[10]:


# model


# In[11]:


model.layer4[-1]


# In[12]:


model.layer1[0]


# ## 选择可解释性分析方法（任选一个）

# In[13]:


from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad


# In[14]:


# Grad-CAM
from pytorch_grad_cam import GradCAM
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)


# In[15]:


# # Grad-CAM++
# from pytorch_grad_cam import GradCAMPlusPlus
# target_layers = [model.layer4[-1]]
# cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)


# ## 生成CAM热力图

# In[16]:


cam_map = cam(input_tensor=input_tensor, targets=targets)[0] # 不加平滑
# cam_map = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0] # 加平滑


# In[17]:


# 也可以这么写
# with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
#     cam_map = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0]


# ## 可视化CAM热力图

# In[18]:


cam_map.shape


# In[19]:


plt.imshow(cam_map)
plt.show()


# In[20]:


import torchcam
from torchcam.utils import overlay_mask

result = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.6) # alpha越小，原图越淡


# In[24]:


result


# In[22]:


result.save('output/B1.jpg')


# In[ ]:





# In[ ]:




