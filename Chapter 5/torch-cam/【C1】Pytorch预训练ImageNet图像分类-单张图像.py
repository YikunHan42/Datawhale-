#!/usr/bin/env python
# coding: utf-8

# # Pytorch预训练ImageNet-1000图像分类-单张图像
# 
# 通过Python API方式，使用torchcam算法库，对Pytorch预训练ImageNet-1000图像分类模型进行基于CAM的可解释性分析。
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-8-19

# ## 导入工具包

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image

import torch
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 导入pillow中文字体

# In[2]:


from PIL import ImageFont, ImageDraw
# 导入中文字体，指定字体大小
font = ImageFont.truetype('SimHei.ttf', 50)


# ## 导入ImageNet预训练模型

# In[3]:


from torchvision.models import resnet18
model = resnet18(pretrained=True).eval().to(device)


# ## 导入可解释性分析方法

# In[4]:


from torchcam.methods import SmoothGradCAMpp 
# CAM GradCAM GradCAMpp ISCAM LayerCAM SSCAM ScoreCAM SmoothGradCAMpp XGradCAM

cam_extractor = SmoothGradCAMpp(model)


# ## 预处理

# In[5]:


from torchvision import transforms
# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# ## 运行图像分类预测

# In[6]:


img_path = 'test_img/cat_dog.jpg'


# In[7]:


img_pil = Image.open(img_path)
input_tensor = test_transform(img_pil).unsqueeze(0).to(device) # 预处理


# In[8]:


input_tensor.shape


# In[9]:


pred_logits = model(input_tensor)
pred_top1 = torch.topk(pred_logits, 1)
pred_id = pred_top1[1].detach().cpu().numpy().squeeze().item()


# In[10]:


pred_id


# ## 生成可解释性分析热力图

# In[11]:


activation_map = cam_extractor(pred_id, pred_logits)


# In[12]:


activation_map = activation_map[0][0].detach().cpu().numpy()


# In[13]:


activation_map.shape


# In[14]:


activation_map


# ## 可视化

# In[15]:


plt.imshow(activation_map)
plt.show()


# In[16]:


from torchcam.utils import overlay_mask

result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)


# In[17]:


result


# ## 整理代码：设置类别、中文类别显示

# 载入ImageNet 1000图像分类标签
# 
# ImageNet 1000类别中文释义：https://github.com/ningbonb/imagenet_classes_chinese

# In[18]:


import pandas as pd
df = pd.read_csv('imagenet_class_index.csv')
idx_to_labels = {}
idx_to_labels_cn = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = row['class']
    idx_to_labels_cn[row['ID']] = row['Chinese']


# In[19]:


# idx_to_labels


# 边牧犬 232
# 
# 牧羊犬 231
# 
# 虎斑猫 282、281

# In[26]:


img_path = 'test_img/cat_dog.jpg'

# 可视化热力图的类别ID，如果为 None，则为置信度最高的预测类别ID
show_class_id = 231
# show_class_id = None

# 是否显示中文类别
Chinese = True
# Chinese = False


# In[27]:


# 前向预测
img_pil = Image.open(img_path)
input_tensor = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
pred_logits = model(input_tensor)
pred_top1 = torch.topk(pred_logits, 1)
pred_id = pred_top1[1].detach().cpu().numpy().squeeze().item()


# In[28]:


# 可视化热力图的类别ID，如果不指定，则为置信度最高的预测类别ID
if show_class_id:
    show_id = show_class_id
else:
    show_id = pred_id
    show_class_id = pred_id


# In[47]:


# 生成可解释性分析热力图
activation_map = cam_extractor(show_id, pred_logits)
activation_map = activation_map[0][0].detach().cpu().numpy()
result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)


# In[48]:


# 在图像上写字
draw = ImageDraw.Draw(result)

if Chinese:
    # 在图像上写中文
    text_pred = 'Pred Class: {}'.format(idx_to_labels_cn[pred_id])
    text_show = 'Show Class: {}'.format(idx_to_labels_cn[show_class_id])
else:
    # 在图像上写英文
    text_pred = 'Pred Class: {}'.format(idx_to_labels[pred_id])
    text_show = 'Show Class: {}'.format(idx_to_labels[show_class_id])
# 文字坐标，中文字符串，字体，rgba颜色
draw.text((50, 100), text_pred, font=font, fill=(255, 0, 0, 1))
draw.text((50, 200), text_show, font=font, fill=(255, 0, 0, 1))


# In[49]:


result


# ## 小练习
# 
# 1.更换不同的可解释性分析算法
# 
# 2.分析不同的层
# 
# 3.更换成你自己手机拍摄的照片

# In[ ]:




