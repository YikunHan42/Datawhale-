#!/usr/bin/env python
# coding: utf-8

# # LayerCAM可解释性分析
# 
# 对单张图像，进行LayerCAM可解释性分析。
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-9-21

# ## 导入工具包

# In[1]:


from torchvision.models import vgg16, resnet50

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 载入ImageNet预训练图像分类模型

# In[2]:


model = vgg16(pretrained=True).eval().to(device)
# model = resnet50(pretrained=True).eval().to(device)


# ## 图像预处理

# In[3]:


from torchvision import transforms

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(224),
                                     # transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# ## 载入测试图像

# In[6]:


img_path = 'test_img/snake.jpg'

# img_path = 'test_img/cat_dog.jpg'


# In[7]:


img_pil = Image.open(img_path)


# In[8]:


# img_pil


# In[9]:


input_tensor = test_transform(img_pil).unsqueeze(0).to(device) # 预处理


# In[10]:


input_tensor.shape


# ## 输入模型，执行前向预测
# 
# 参考教程：https://www.bilibili.com/video/BV1qe4y1D7zD

# In[11]:


# 执行前向预测，得到所有类别的 logit 预测分数
pred_logits = model(input_tensor) 

import torch.nn.functional as F
pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算


# In[12]:


pred_softmax.shape


# ## 获得图像分类预测结果

# In[13]:


n = 5
top_n = torch.topk(pred_softmax, n)


# In[14]:


top_n


# In[15]:


# 解析出类别
pred_ids = top_n[1].cpu().detach().numpy().squeeze()


# In[16]:


pred_ids


# In[17]:


# 解析出置信度
confs = top_n[0].cpu().detach().numpy().squeeze()


# In[18]:


confs


# In[19]:


# 载入ImageNet 1000图像分类标签
df = pd.read_csv('imagenet_class_index.csv')
idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = [row['wordnet'], row['class']]


# In[20]:


for i in range(n):
    class_name = idx_to_labels[pred_ids[i]][1] # 获取类别名称
    confidence = confs[i] * 100 # 获取置信度
    text = '{:<5} {:<15} {:>.4f}'.format(pred_ids[i], class_name, confidence)
    print(text)


# ## 指定分析的类别

# In[21]:


from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 如果 targets 为 None，则默认为最高置信度类别
targets = [ClassifierOutputTarget(56)]


# ## 分析模型结构，确定待分析的层

# In[22]:


# model


# ## 选择可解释性分析方法

# In[23]:


# LayerCAM
from pytorch_grad_cam import LayerCAM
target_layers = [model.features[8]] # vgg16
# target_layers = [model.layer3[0]] # resnet50
cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=True)


# ## 生成CAM热力图

# In[24]:


cam_map = cam(input_tensor=input_tensor, targets=targets)[0] # 不加平滑


# ## 可视化CAM热力图

# In[25]:


cam_map.shape


# In[26]:


cam_map.dtype


# In[27]:


plt.imshow(cam_map)
plt.show()


# In[28]:


import torchcam
from torchcam.utils import overlay_mask

result = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.12) # alpha越小，原图越淡


# In[32]:


# result


# In[30]:


result.save('output/B2.jpg')


# ## 思考题
# 
# 1.更换从浅至深的不同层，观察LayerCAM的变化，效果在变得更好吗？
# 
# 2.把模型换成resnet50，LayerCAM热力图会出现“棋盘网格”现象，思考原因及改进方法。
# 
# 3.换一张包含多类物体的图片，查看不同层、不同类别的LayerCAM热力图，是否符合class-discriminative？

# In[196]:


# 棋盘网格现象


# In[ ]:





# In[ ]:




