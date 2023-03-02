#!/usr/bin/env python
# coding: utf-8

# # Pytorch-预训练ImageNet图像分类可解释性分析
# 
# 对Pytorch模型库中的ImageNet预训练图像分类模型进行可解释性分析。可视化指定预测类别的shap值热力图。
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
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import shap

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 载入ImageNet预训练图像分类模型

# In[2]:


model = torchvision.models.mobilenet_v2(pretrained=True, progress=False).eval().to(device)


# ## 载入ImageNet1000类别标注名称

# In[6]:


with open('data/imagenet_class_index.json') as file:
    class_names = [v[1] for v in json.load(file).values()]


# In[7]:


# class_names


# ## 载入一张测试图像，整理维度

# In[16]:


# img_path = 'test_img/test_草莓.jpg'

img_path = 'test_img/cat_dog.jpg'


# In[22]:


img_pil = Image.open(img_path)
X = torch.Tensor(np.array(img_pil)).unsqueeze(0)


# In[23]:


X.shape


# ## 预处理

# In[24]:


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x 
        

transform= [
    transforms.Lambda(nhwc_to_nchw),
    transforms.Resize(224),
    transforms.Lambda(lambda x: x*(1/255)),
    transforms.Normalize(mean=mean, std=std),
    transforms.Lambda(nchw_to_nhwc),
]

inv_transform= [
    transforms.Lambda(nhwc_to_nchw),
    transforms.Normalize(
        mean = (-1 * np.array(mean) / np.array(std)).tolist(),
        std = (1 / np.array(std)).tolist()
    ),
    transforms.Lambda(nchw_to_nhwc),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)


# ## 构建模型预测函数

# In[25]:


def predict(img: np.ndarray) -> torch.Tensor:
    img = nhwc_to_nchw(torch.Tensor(img)).to(device)
    output = model(img)
    return output


# In[26]:


def predict(img):
    img = nhwc_to_nchw(torch.Tensor(img)).to(device)
    output = model(img)
    return output


# ## 测试整个工作流正常

# In[37]:


Xtr = transform(X)
out = predict(Xtr[0:1])


# In[39]:


out.shape


# In[40]:


classes = torch.argmax(out, axis=1).detach().cpu().numpy()
print(f'Classes: {classes}: {np.array(class_names)[classes]}')


# ## 设置shap可解释性分析算法

# In[46]:


# 构造输入图像
input_img = Xtr[0].unsqueeze(0)


# In[44]:


input_img.shape


# In[131]:


batch_size = 50

n_evals = 5000 # 迭代次数越大，显著性分析粒度越精细，计算消耗时间越长

# 定义 mask，遮盖输入图像上的局部区域
masker_blur = shap.maskers.Image("blur(64, 64)", Xtr[0].shape)

# 创建可解释分析算法
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)


# ## 指定单个预测类别

# In[132]:


# 281：虎斑猫 tabby
shap_values = explainer(input_img, max_evals=n_evals, batch_size=batch_size, outputs=[281])


# In[133]:


# 整理张量维度
shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0] # 原图
shap_values.values = [val for val in np.moveaxis(shap_values.values[0],-1, 0)] # shap值热力图


# In[134]:


# 原图
shap_values.data.shape


# In[135]:


# shap值热力图
shap_values.values[0].shape


# In[162]:


# 可视化
shap.image_plot(shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=shap_values.output_names)


# ## 指定多个预测类别

# In[153]:


# 232 边牧犬 border collie
# 281：虎斑猫 tabby
# 852 网球 tennis ball
# 288 豹子 leopard
shap_values = explainer(input_img, max_evals=n_evals, batch_size=batch_size, outputs=[232, 281, 852, 288])


# In[154]:


# 整理张量维度
shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0] # 原图
shap_values.values = [val for val in np.moveaxis(shap_values.values[0],-1, 0)] # shap值热力图


# In[155]:


# shap值热力图：每个像素，对于每个类别的shap值
shap_values.shape


# In[156]:


# 可视化
shap.image_plot(shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=shap_values.output_names)


# ## 前k个预测类别

# In[141]:


topk = 5


# In[142]:


shap_values = explainer(input_img, max_evals=n_evals, batch_size=batch_size, outputs=shap.Explanation.argsort.flip[:topk])


# In[143]:


# shap值热力图：每个像素，对于每个类别的shap值
shap_values.shape


# In[144]:


# 整理张量维度
shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0] # 原图
shap_values.values = [val for val in np.moveaxis(shap_values.values[0],-1, 0)] # 各个类别的shap值热力图


# In[146]:


# 各个类别的shap值热力图
len(shap_values.values)


# In[147]:


# 第一个类别，shap值热力图
shap_values.values[0].shape


# In[148]:


# 可视化
shap.image_plot(shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=shap_values.output_names
                )


# In[ ]:




