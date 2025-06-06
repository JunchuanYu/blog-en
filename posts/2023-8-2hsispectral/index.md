---
title: "光谱库读写+光谱处理及交互式采集"
author: 
  - "于峻川"
date: "2023-8-2"
categories:
  - Posts
  - Hyperspectral
  - Teaching
  - Workshop
image: "welcome.jpg"
toc: true
---
# 光谱库读写+光谱处理及交互式采集
![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/WRSDP-4.png)


<p align="justify ">上一篇我们介绍了利用Python对不同格式的高光谱数据进行读写，本章将介绍如何从图像中交互式的采集光谱，sli格式光谱库的读写方式以及一阶微分、光谱平滑、多项式拟合、吸收深度计算等光谱处理方法。<p>

-  交互式工具的制作
-  光谱库文件读写，头文件编辑等
-  光谱一阶微分、平滑、多项式拟合、吸收深度计算等处理


```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os,glob,time
import spectral.io.envi as envi
import importlib
from utils import * #之前分享过的function都放到了utils里，此文档不在展示，文末回复关键字获取
import sys,importlib
importlib.reload(sys.modules['utils'])# 在需要重新加载的地方调用 reload() 函数
# !pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension
```




    <module 'utils' from 'x:\\WRSDP\\Second Week\\TASK2\\utils.py'>



### 1. 光谱数据处理


```python
# 用sepctral库来加载sli光谱库数据
Cuprite_em=envi.open('./data/cuprite_end.hdr') # 读取cuprite端元光谱库
spec_data=Cuprite_em.spectra.transpose()
spec_meta = envi.read_envi_header('./data/cuprite_end.hdr') #读取光谱库的头文件
spec_name=spec_meta['spectra names']
spec_wl=spec_meta['wavelengths']
show_spectral(spec_data,spec_name,'Spectral Signature of Endmembers')
```


![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/task2-2_5_0.png)

   


### 2. 光谱分析

光谱分析方法有很多，这里展示常用的一些方法


```python
# 吸收深度计算
from scipy.signal import *

def depthcal(band,spec_wl,l, r,x=None):
    if x==None:
        x=np.argmin(band[l:r])+l
    xwl, lwl, rwl=(spec_wl[x], spec_wl[l], spec_wl[r])
    a = 1 - (rwl - xwl) / (xwl - lwl)
    b = (rwl - xwl) / (xwl - lwl)
    result = 1 - band[x] / (a * band[l] + b * band[r])
    return result

mus_wl =np.array( [float(x) for x in spec_wl])
mul_data=spec_data[:,6]
l=145
r=165
x=np.argmin(mul_data[l:r])+l
specdepth=depthcal(mul_data,mus_wl,l,r)
print(x,specdepth)

# 利用scipy光谱平滑和微分运算
smooth_spec = savgol_filter(spec_data[:,6], window_length=25, polyorder=7)
spec_data_diff = np.diff(spec_data[:,6], axis=0)
```

    157 0.28345667940108465
    


```python
# 结果可视化展示，为了方便对比，将平滑曲线与一阶微分曲线就做了y轴数值的偏移处理
fig = plt.figure(figsize=(7, 4))
plt.style.use('seaborn')
plt.plot(smooth_spec-0.2, label='smooth spectrum', color='black')
plt.plot(spec_data_diff+0.2, label='diff spectrum', color='orange')
plt.plot(spec_data[:,6], label='original spectrum', color='darkgreen')
plt.plot(range(l,r), spec_data[l:r,6], 'blue', label='interval spectrum') 
plt.fill_between(range(l,r),np.max(spec_data),  color='lightblue', alpha=0.3)
plt.axvline(x=x, color='r', linestyle='--',label='absorption position')
plt.xlabel('Bands', fontweight='bold', fontname='Times New Roman', fontsize=12)
plt.ylabel('Reflectance', fontweight='bold', fontname='Times New Roman', fontsize=12)
plt.title('Diagnostic Spectral Features')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
```



![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/task2-2_9_1.png)
    



```python
# 多项式拟合光谱曲线

from scipy.optimize import curve_fit

def fit_func(x, a, b, c):
    return a * x**2 + b * x + c
params, params_cov = curve_fit(fit_func, np.arange(len(spec_wl)), spec_data[:,8])
pred=fit_func(np.arange(len(spec_wl)), *params)

fig = plt.figure(figsize=(7, 4))
plt.style.use('seaborn')
plt.plot(spec_data[:,8], label='original spectrum', color='darkgreen')
plt.plot(pred, 'orange', label='fitted spectrum') 
plt.xlabel('Bands', fontweight='bold', fontname='Times New Roman', fontsize=12)
plt.ylabel('Reflectance', fontweight='bold', fontname='Times New Roman', fontsize=12)
plt.title('Curve Fitting')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
```

    
![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/task2-2_10_1.png)



### 3. 影像光谱交互采集


```python
# 影像光谱采集
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display,HTML
import ipywidgets as widgets
%matplotlib widget

def display_image_with_mouse_click(image,rgb):
    # 存储点的坐标和通道像元值
    selected_points = []
    color_strings = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896']

    def on_click(trace, points, state):
        nonlocal selected_points

        if points.point_inds:
            x_index = points.xs[0]
            y_index = points.ys[0]

            # 添加一个散点图
            fig.add_trace(go.Scatter(
                x=[x_index],  # 点的横坐标
                y=[y_index],  # 点的纵坐标
                mode='markers',
                marker=dict(size=10, color=color_strings[len(selected_points)], opacity=1),
            ))
            fig.update()

            # 记录点的坐标和通道像元值
            x_value = int(x_index)
            y_value = int(y_index)
            pixel_values = image[y_value, x_value]
            selected_points.append({'x': x_value, 'y': y_value, 'pixel_values': pixel_values})
            # 更新多波段像元值曲线
            display_pixel_values(fig_pixel_values, selected_points, rgb, color_strings)

    # 绘制像元值曲线
    def display_pixel_values(fig, points, rgb, colors):
        if len(points) == 0:
            return

        for i, point in enumerate(points):
            if len(fig.data) > i:
                # 如果曲线已存在，则更新数据
                fig.data[i].x = np.arange(len(point['pixel_values']))
                fig.data[i].y = np.array(point['pixel_values'])
                fig.data[i].marker.color = colors[i]
            else:
                # 如果曲线不存在，则添加新的曲线
                fig.add_trace(go.Scatter(
                    x=np.arange(len(point['pixel_values'])),
                    y=np.array(point['pixel_values']),
                    mode='lines',
                    name='Point {}'.format(i+1),
                    marker=dict(color=colors[i])
                ))
        fig.update()
    rows, cols, channels = image.shape

    fig = go.FigureWidget(data=go.Image(z=rgb, colormodel='rgb'))
    fig.update_layout(title='Image',
                      xaxis=dict(title='Columns'),
                      yaxis=dict(title='Rows'),
                      clickmode='event+select')
    fig_pixel_values = go.FigureWidget()
    fig_pixel_values.update_layout(title='Spectrum Collection',
                          xaxis=dict(title='Band'),
                          yaxis=dict(title='Reflectance'))
    aspect_ratio = cols / rows  # 宽高比
    if aspect_ratio > 1:
        fig.update_layout(height=500, width=int(500 * aspect_ratio))
    else:
        fig.update_layout(height=int(500 / aspect_ratio), width=500)

    fig.data[0].on_click(on_click)
    # 设置鼠标悬停时显示的坐标信息
    fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y}')


    fig_pixel_values.update_layout(height=500, width=800)

    # 将FigureWidget放入HBox布局中
    box_layout = widgets.Layout(display='flex', flex_flow='row', justify_content='space-between')
    hbox = widgets.HBox([fig, fig_pixel_values], layout=box_layout)
    display(hbox)
    return selected_points
```


```python
# 定义读取文件
def read_tiff(file):
    img_arr,meta = Load_image_by_Gdal(file)
    if meta['img_bands'] > 1:
        img_arr=img_arr.transpose(( 1, 2,0))
    return img_arr,meta

filepath='./data/Cuprite_ref188.img'
img_arr,img_meta = read_tiff(filepath)
print(img_arr.shape,img_meta.keys())

```

    (500, 500, 188) dict_keys(['img_bands', 'geomatrix', 'projection', 'wavelengths', 'wavelength_units'])
    


```python
# 对图像进行增强处理
rgb=linear_stretch(img_arr[:,:,[27,18,9]],True)*255
selected_points = display_image_with_mouse_click(img_arr,rgb)
# 通过交互式选点，可以更准确的获取影像光谱数据，图片可以放大缩小
%matplotlib inline
```


    HBox(children=(FigureWidget({
        'data': [{'colormodel': 'rgb',
                  'hovertemplate': 'X: %{x}<br>Y:…

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/20230802_090314.gif)

### 4. 光谱数据保存为光谱库


```python
# 采集的光谱数据进行保存
spectrals=np.array([point['pixel_values'] for point in selected_points])
print(spectrals.shape)
show_spectral(spectrals.T)

```


```python
# 与刚刚光谱库中获取的个端元光谱进行整合，输出为一个样本库集合，注意不同的数据格式不统一，需要进行转换
newspec = np.stack((spec_data[:,6],smooth_spec-0.2,pred), axis=1)
newspec_lib=np.concatenate((newspec*10000,spectrals.T),axis=-1)
show_spectral(newspec_lib)

```

  


    
![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/task2-2_19_1.png)    
    



```python

spec_name=['original','smooth','fitting','select1','select2','select3','select4']

em_meta={}
em_meta['wavelengths']=spec_wl
em_meta['wavelength units']='Micrometers'
em_meta['spectra names'] = spec_name

spec=envi.SpectralLibrary(newspec_lib.transpose(), em_meta)
spec.save('./data/new_cuprite_end')

```


---------------------------
请关注微信公众号【45度科研人】获取更多精彩内容，欢迎后台留言！

<span style="display: block; text-align: center; margin-left: auto; margin-right: auto;">
    <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/wechat-simple.png" width="300"  alt="">
</span>