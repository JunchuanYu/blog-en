---
title: "高光谱数据去噪及降维处理"
author: 
  - "Junchuan Yu"
date: "2023-8-11"
categories:
  - Posts
  - Hyperspectral
  - Teaching
  - Workshop
image: "welcome.png"
toc: true
---

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/WRSDP-4-3.png)

# 高光谱数据去噪及降维处理

<p align="justify ">上一篇我们介绍了如何在影像中通过交互的方式采集光谱，并对光谱库文件进行读写，接下来将介绍光谱去噪和降维处理的一些传统方法。事实上在高光谱图像处理方面深度学习技术提供了更好的解决方案，后面我们在谈人工智能遥感应用的时候再一一介绍，本篇主要涉及异常值剔除、利用距匹配(Moment matching)进行去噪处理，以及主成分分析(PCA,principal component analysis )和最小噪声分离(MNF,Minimum Noise Fraction)来进行高光谱降维处理。<p>

-  异常值剔除、线性拉伸
-  Moment matching去噪处理
-  PCA、MNF变换等



## 1 异常值剔除及拉伸显示


```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os,glob,time
import importlib
from pysptools.noise import Whiten
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from utils import read_tiff,plot_2_img,get_hsi_by_wl,mnf_r,Write_Tiff #之前分享过的function都放到了utils里，此文档不再展示

```


```python
filepath='./data/gaofen-5.img'
img_arr,img_meta = read_tiff(filepath)
print(img_arr.shape,img_meta.keys())
```

    (500, 400, 295) dict_keys(['img_bands', 'geomatrix', 'projection', 'wavelengths', 'wavelength_units'])
    


```python
def stretch_2percent(data):
    stretched_data = np.zeros(data.shape)
    for i in range(data.shape[2]):
        min_val = np.percentile(data[:, :, i], 2)
        max_val = np.percentile(data[:, :, i], 98)
        stretched_data[:, :, i] = (data[:, :, i] - min_val) / (max_val - min_val)
    stretched_data = np.clip(stretched_data, 0, 1)
    return stretched_data
def get_hsi_by_band(hsi_img, bands_to_get):
	bands=np.array(bands_to_get)
	hsi_out = hsi_img[:,:,[int(m) for m in bands]]
	return np.array(hsi_out)
def get_rgb_array(hsi_img,rgblist,wavelengths=None,liner_trans=False):
    n_row, n_col, n_band = hsi_img.shape
    if wavelengths is not None:
        red_wavelengths = list(range(619,659))
        green_wavelengths = list(range(549,570))
        blue_wavelengths = list(range(449,495))
        RGB_img = np.zeros((n_row, n_col, 3))
        RGB_img[:,:,0] = np.mean(get_hsi_by_wl(hsi_img, wavelengths, red_wavelengths), axis=2)
        RGB_img[:,:,1] = np.mean(get_hsi_by_wl(hsi_img, wavelengths, green_wavelengths), axis=2)
        RGB_img[:,:,2] = np.mean(get_hsi_by_wl(hsi_img, wavelengths, blue_wavelengths), axis=2)
    else:
        RGB_img =get_hsi_by_band(hsi_img, rgblist)
    if liner_trans:
        RGB_img =stretch_2percent(RGB_img)
    return RGB_img
```


```python
wl=np.array(img_meta['wavelengths'], dtype=np.float32) 
rgb1=get_hsi_by_band(img_arr,[59,38,17])
rgb2=get_rgb_array(img_arr,[59,38,17],None,liner_trans=True)
plot_2_img(rgb1/np.max(rgb1),rgb2/np.max(rgb2),'Original','2% Linear stretch')
```



![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/task2-3_8_0.png)

   


## 2 距匹配降噪处理

Moment Matching 是一种常用的图像去噪方法，其基本原理是根据图像的统计特性来对图像进行修正，以降低噪声的影响。首先计算图像的各个波段的均值和标准差，通过比较不同行列之间的差异，可以推断出噪声所引起的扰动。然后，通过调整图像的增益和偏移量，使得匹配后的均值和标准差能够更好地反映噪声的统计特性。通过 Moment Matching 方法，我们可以使图像恢复到更接近真实场景的状态，减少噪声对图像质量的影响。需要注意的是，Moment Matching 可能会引入一定的偏差，特别是对于强烈的噪声或者图像数值、为例差异较大的的情况。因此，在应用 Moment Matching 进行图像去噪时，需要考虑下垫面的组成和特点。

   - 增益：$\text{gain} = \frac{\text{band\_std}}{\text{ns\_std}}$
   - 偏移量：$\text{offset} = \text{bands\_avg} - \text{gain} \times \text{ns\_avg}$
   - 去噪处理：$\text{final\_image}[i, j, b] = \text{data}[i, j, b] \times \text{gain}[b] + \text{offset}[b]$

上述公式中，$\text{bands\_avg}$ 表示各个波段的像素均值，$\text{band\_std}$ 表示各个波段的像素标准差。$\text{ns\_avg}$ 表示b波段各个列的像素均值，$\text{ns\_std}$ 表示b波段各个列的像素标准差。$\text{gain}$ 表示增益，$\text{offset}$ 表示偏移量。$\text{final\_image}$ 是去噪后的最终图像。


```python
def moment_matching(data):
    rows, cols, bands=data.shape
    final_image = np.zeros([rows, cols,bands])
    bands_avg = np.mean(data, axis=(0,1))
    band_std = np.std(data, axis=(0,1))
    # print(bands_avg.shape,band_std.shape)
    ns_std = np.std(data, axis=(0,))  
    ns_avg = np.mean(data, axis=(0,))
    # print(ns_std.shape,ns_avg.shape)
    gain =  np.broadcast_to(band_std, ns_std.shape)/ np.where(ns_std == 0, 1e-8, ns_std)
    offset = np.broadcast_to(bands_avg, ns_avg.shape) - gain * ns_avg
    for i in range(bands):
        for j in range(cols):
            final_image[:,j,i] = data[:,j,i] * gain[j,i] + offset[j,i]
    return final_image

```


```python
denoised_img=moment_matching(img_arr)
i=-1
plot_2_img(img_arr[:,:,i]/6000,denoised_img[:,:,i]/6000,"Original", "Denoised data",cmap1='gray_r',cmap2='gray_r')
```



![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/task2-3_12_0.png)    
    


## 3 降维处理

### 3.1 主成分分析

遥感影像的主成分分析（Principal Component Analysis, PCA）是一种常用的降维技术，用于提取遥感影像中的主要信息。它通过线性变换将多波段影像转换为新的特征空间，使得在新空间中的样本间的协方差最小化，可以在降低数据维度的同时保留尽可能多的信息。
。以下是遥感影像主成分分析的原理、主要过程和计算公式：

**主要过程：**

1. 数据标准化：对原始图像进行均值中心化和标准差归一化，使不同波段之间具有相同的尺度
   - $\mathbf{X_i} = \frac{\mathbf{X_i} - \mu_i}{\sigma_i}$

     其中，$\mathbf{X_i}$ 是第 $i$ 个波段的图像，$\mu_i$ 是第 $i$ 个波段的均值，$\sigma_i$ 是第 $i$ 个波段的标准差。
2. 协方差矩阵计算：计算标准化后的图像数据的协方差矩阵，用于度量不同波段之间的相关性
   - $\mathbf{C} = \frac{1}{N-1} \sum_{i=1}^{N}(\mathbf{X_i}-\boldsymbol{\mu})(\mathbf{X_i}-\boldsymbol{\mu})^T$

     其中，$\mathbf{C}$ 是协方差矩阵，$N$ 是波段数，$\mathbf{X_i}$ 是第 $i$ 个波段的标准化图像，$\boldsymbol{\mu}$ 是均值向量。
3. 特征值分解：对协方差矩阵进行特征值分解，得到特征向量和特征值。特征向量代表了原始数据的主要方向，特征值表示数据在这些方向上的重要程度
   - $\mathbf{C} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T$

     其中，$\mathbf{U}$ 是特征向量矩阵，$\boldsymbol{\Lambda}$ 是对角特征值矩阵。
4. 选择主成分：选择特征值最大的前 $k$ 个特征向量作为主成分，其中 $k$ 是期望的降维后的维度。这些主成分按照特征值的降序排列
   - $\mathbf{W} = \mathbf{U}(:, 1:k)$

     其中，$\mathbf{W}$ 是前 $k$ 个最大特征值对应的特征向量组成的矩阵。
5. 变换到新的特征空间：将原始数据与选择的主成分相乘，得到在新的特征空间中的表示
   - $\mathbf{F} = \mathbf{X} \mathbf{W}$
   
     其中，$\mathbf{F}$ 是新的特征空间图像，$\mathbf{X}$ 是原始标准化后的图像数据。




```python
n_components = 50
pca = PCA(n_components)
principalComponents = pca.fit_transform(img_arr.reshape(-1,img_arr.shape[-1]))
img_pc=principalComponents.reshape(img_arr.shape[:-1]+(n_components,))
print(img_pc.shape)
```

    (500, 400, 50)
    


```python
ev=pca.explained_variance_ratio_
fig = plt.figure(figsize=(8, 4))
plt.style.use('seaborn')
plt.plot(np.cumsum(ev))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
```


![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/task2-3_17_1.png)
    



```python
def norm_data(data):
    scaler = MinMaxScaler()
    normalized_data_2d = scaler.fit_transform(data.reshape(-1, data.shape[-1]))
    normalized_data = normalized_data_2d.reshape(data.shape)
    return normalized_data
```


```python
normalized_data=norm_data(img_pc[:,:,:4])
fig, ax = plt.subplots(1,4, figsize=(10, 4))
for i in range(4):
    ax[ i].imshow(normalized_data[:, :, i], cmap='Set3')
    ax[ i].set_title('Band {}'.format(i+1))
    ax[ i].axis('off')
plt.tight_layout()  # 自动调整子图之间的间距
plt.show()
```


![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/task2-3_19_0.png)   
    


### 3.2 MNF变换

最小噪声分离（Minimum Noise Fraction, MNF）是一种常用于遥感图像处理的特征提取方法。它通过将多波段图像转换为新的特征空间，将信号与噪声分离开来，以提高遥感图像的解译质量，有助于后续的分类、目标检测和监督分类等任务。MNF变换和主成分分析都能够实现高光谱数据的降维，且计算过程有许多相似之处，二者主要区别如下：
- 目标：MNF变换旨在将遥感图像中的噪声和信号有效分离，并提取有效的信号成分。主成分分析旨在找到原始数据中的主要方向，并对主要方向进行重要性排序。
- 特征选择：在MNF变换中，特征向量是根据与噪声相关性的大小来选择的，以获得较低噪声的特征。而在主成分分析中，特征向量是根据其对应的特征值的大小来选择的，以表示数据中的主要方向。
- 协方差矩阵：在MNF变换中，计算的是协方差矩阵，用于描述图像数据之间的相关性和噪声特征。主成分分析也需要计算协方差矩阵，但其目的是为了找到数据的主要方向。
- 结果表达：MNF变换得到的结果是一组新的特征空间图像，其中每个特征都具有较低的噪声。主成分分析得到的结果是一组新的主成分，按照其对应的特征值的大小进行排序。


```python
def bad_band_index(data, wl, wl_range1,wlrange2):
    excluded_range_1 = np.logical_and(wl >= wl_range1[0], wl <= wl_range1[1])
    excluded_range_2 = np.logical_and(wl >= wlrange2[0], wl <= wlrange2[1])
    included_bands = np.logical_not(np.logical_or(excluded_range_1, excluded_range_2))
    band_indices = np.where(included_bands)[0]
    return band_indices

def MNF_func(data,n_components = 15):
    num_dimensions = len(data.shape)
    if num_dimensions == 2:
        data = np.expand_dims(data, axis=0)
    #check data shape
    if data.shape[-1] <=n_components:
        raise ValueError("For MNF,n_components must be smaller than spectral channels")
    mnf_result = mnf_r(data, n_components)
    # Return result in dimensionality of input
    if num_dimensions == 2:
        return np.squeeze(mnf_result)
    else:
        return mnf_result    
```


```python
band_indices=bad_band_index(denoised_img,wl,[1200,1500],[1700,2000])
clean_data,newwl=img_arr[:,:,band_indices],wl[band_indices]
mnf_trans=MNF_func(clean_data,15)
reduced_data=norm_data(mnf_trans[:,:,:3])
```


```python
plot_2_img(rgb2/np.max(rgb2),reduced_data,"Original_data", "First 3 components in RGB")
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


 ![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/task2-3_24_1.png)

   



```python
Write_Tiff(np.transpose(mnf_trans,(2,0,1)), './data/reduced.tiff', img_meta['geomatrix'], img_meta['projection'],  wavelengths=None, wavelength_units=None)
```

    missing geomatrix or projection information.
    

---------------------------
请关注微信公众号【45度科研人】获取更多精彩内容，欢迎后台留言！

<span style="display: block; text-align: center; margin-left: auto; margin-right: auto;">
    <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/wechat-simple.png" width="300"  alt="">
</span>