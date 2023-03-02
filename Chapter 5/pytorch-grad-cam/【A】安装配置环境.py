#!/usr/bin/env python
# coding: utf-8

# # 安装配置环境
# 
# 安装配置 pytorch-grad-cam 环境
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-9-20

# ## 安装两个代码包

# In[1]:


get_ipython().system('pip install grad-cam torchcam')


# ## 下载 pytorch-grad-cam

# In[14]:


get_ipython().system('git clone https://github.com/jacobgil/pytorch-grad-cam.git')


# ## 创建目录

# In[8]:


import os


# In[9]:


# 存放测试图片
os.mkdir('test_img')

# 存放结果文件
os.mkdir('output')

# 存放模型权重文件
os.mkdir('checkpoint')


# ## 下载动物测试图片至`test_img`目录

# In[10]:


get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220919-explain/test_img/puppies.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220919-explain/test_img/bear.jpeg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220919-explain/test_img/box_tabby.png -P test_img')

# 蛇，来源：https://www.pexels.com/zh-cn/photo/80474/
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220919-explain/test_img/snake.jpg -P test_img')

# 长颈鹿和斑马，来源：https://www.istockphoto.com/hk/%E7%85%A7%E7%89%87/giraffes-and-zebras-at-waterhole-gm503592172-82598465
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220919-explain/test_img/giraffe_zebra.jpg -P test_img')

# 大象、狮子、羚羊，来源：https://www.istockphoto.com/hk/%E7%85%A7%E7%89%87/%E5%A4%A7%E8%B1%A1%E5%92%8C%E7%8D%85%E5%AD%90-gm1136053333-30244130
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220919-explain/test_img/africa.jpg -P test_img')

# 边牧犬，来源：https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/border-collie.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/cat_dog.jpg -P test_img')


# ## 下载水果测试图片至`test_img`目录

# In[11]:


get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_fruits.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_orange_2.jpg -P test_img ')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_bananan.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_kiwi.jpg -P test_img')

# 草莓图像，来源：https://www.pexels.com/zh-cn/photo/4828489/
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/0818/test_草莓.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_石榴.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_orange.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_lemon.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_火龙果.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/watermelon1.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/banana1.jpg -P test_img')


# ## 下载 ImageNet 1000类别信息

# In[12]:


get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/meta_data/imagenet_class_index.csv')


# ## 自己训练得到的30类水果图像分类模型

# In[13]:


# 下载样例模型文件
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/checkpoints/fruit30_pytorch_20220814.pth -P checkpoint')

# 下载 类别名称 和 ID索引号 的映射字典
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/fruit30/labels_to_idx.npy')
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/fruit30/idx_to_labels.npy')


# ## 验证安装配置成功

# In[15]:


import pytorch_grad_cam


# In[ ]:




