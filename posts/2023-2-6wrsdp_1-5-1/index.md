---
title: ' 基于时间序列遥感的水稻类型识别-P1 '
subtitle: 'Work with Remote Sensing Data in Python: Lesson 1-5-1'
author:
  - 于峻川
date: '2023-2-6'
categories:
  - Posts
  - Deep leanring
  - Teaching
  - Workshop
  - GEE
image: welcomenew.jpg
toc: true
---

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/WRSDP-5-1.jpg#pic_center)


<p align="justify ">Goole Earth Engine (GEE)，GEE相信大家都有一定了解，这是一款谷歌推出的云计算平台，提供了大量开源的遥感数据，支持在线用JAVA语言或调用Pyhton API接口实现遥感应用（需要科学上网）。在[上一篇]分享中,我们利用GEE结合哨兵1号雷达数据实现对水体的变化监测，本篇我们继续探索GEE在时序遥感中的应用。下一篇将分享用相同的数据如何在本地用Python实现同样的操作。想要进一步了解GEE的伙伴这里提供了一些资料：<p>

* [GEE安装与配置方法](https://zhuanlan.zhihu.com/p/550679685)
* [GEE主页](https://code.earthengine.google.com/)
* [GEE官方帮助](https://developers.google.com/earth-engine/)
* [GEEMAP主页](https://geemap.org/)
* [GEE知乎大V无形的风](https://zhuanlan.zhihu.com/p/29000495)

# 基于时间序列遥感的水稻类型识别-P1

<p align="justify ">地物的光谱信息是遥感数据的重要特征，对遥感光谱信息的利用经历了从全色影像到多光谱、高光谱再到时间序列的发展历程。近年来，随着卫星遥感的发展和历史数据的积累，获取了大量的重复观测数据。长时序的遥感数据包含光谱维、时间维和空间维四个维度的信息，能够在一定程度上避免“同谱异物”、“同物异谱”的现象，在地物分类、变化检测等方面有广泛应用。本案例利用GEE平台获取2022年度的哨兵二号长时间序列数据构建NDVI时序立方体，利用随机森林算法实现对研究区双季水稻和单季水稻的分类提取。<p>

### 1. 时序立方体构建


```python
import geemap #gee的安装方法见上面链接
import ee
import geemap.colormaps as cm
# ee.Authenticate() #第一次运行不需要注释这句，授权过之后，如果重启kernel这句可以注释掉，不必进行授权步骤也可以，如果报错需要重新授权。
geemap.set_proxy(port=25378) #这里请填入自己计算机的端口号
ee.Initialize()
from matplotlib import pyplot as plt
import numpy as np
```


```python
## 定义示范区范围
region=ee.Geometry.Polygon([[[115.97367, 28.92437],  [117.10013, 28.92824],  [117.09920, 27.93711],   [115.98318, 27.93339],   [115.97367, 28.92437]]])
Map = geemap.Map()
# Map.centerObject(region,zoom=9)
Map.addLayer(region, {}, 'region')
Map
```


    Map(center=[20, 0], controls=(WidgetControl(options=['position', 'transparent_bg'], widget=HBox(children=(Togg…


![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/1.JPG)

定义一些必要的函数


```python
## 哨兵2利用qa波段进行去云的方法
def rmCloudByQA(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11 
    mask =qa.bitwiseAnd(cloudBitMask).eq(0)and(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).toFloat().divide(1e4).copyProperties(image, ["system:time_start"])
## 计算NDVI
def createNDVI(image):
  ndvi = image.normalizedDifference(["B8","B4"]).rename('NDVI').add(ee.Number(1)).divide(ee.Number(2.0)).multiply(ee.Number(255)).toByte()
  return image.addBands(ndvi)
## 将日期作为属性添加到新数据中
def addDate(image):
    date = ee.Date(image.get('system:time_start')).format('yyyyMMdd')
    return image.set("image_date",date) # set的参数为字典
```


```python
## 构建3月-12月的NDVI时序数据集，该区域检索到28景数据
s2Img = ee.ImageCollection('COPERNICUS/S2_SR')
NDVI = s2Img.filterDate('2022-03-01', '2022-11-30').filterBounds(region.centroid()).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 64)).map(rmCloudByQA).map(addDate).map(createNDVI).select('NDVI').sort("image_date")
NDVI=NDVI.filterMetadata('image_date','not_equals','20220730') #剔除一些受云亮影响数据缺失较多的数据
NDVI=NDVI.filterMetadata('image_date','not_equals','20220923')

rgb=s2Img.filterDate('2022-03-01', '2022-11-30').filterBounds(region.centroid()).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)).map(rmCloudByQA).mosaic()
print(NDVI.size().getInfo())
```

    28
    


```python
##将包含28影像的collection转变为具有28个波段的影像
bandname=NDVI.aggregate_array('image_date')
singleimg=NDVI.toBands().rename(bandname) 
## 根据单季稻和双季稻的季节性特点，选择9/4/10几个月份的NDVI作为RGB显示，可大致看出不同时段水稻的分布情况，见后面分析
vis = {'min': 128,
       'max': 230,
       'gamma':2,
       'bands': ['20220918', '20220421', '20221023']}
Map.addLayer(singleimg, vis, "rgb")
Map.addLayer(rgb, {'min': 0, 'max':0.4, 'gamma':1,'bands': ['B4', 'B3', 'B2']}, "s2")
Map
```


    Map(center=[20, 0], controls=(WidgetControl(options=['position', 'transparent_bg'], widget=HBox(children=(Togg…


### 2. 光谱特征分析

研究区双季稻生长期安排因受气候条件的制约相对固定, 大致从每年3 月下旬到10月下旬, 而单季稻生长期安排相对自由, 且全生育期略长, 一般从5月中上旬到10月上旬。下图展示了年内不同熟制水稻武侯历，A为早稻、B为晚稻、C为单季稻，参考[1](http://www.jnr.ac.cn/CN/10.11849/zrzyxb.2011.02.002)、[2](https://www.cnblogs.com/enviidl/p/16943528.html)。

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/10.png)

图中绿色部分为水稻生长旺盛期，对应DNVI较高，因此以9月中旬，4月下旬及10月构建假彩色影像可大致区分单双季水稻。
![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/2.JPG)

<p align="justify ">借助geemap的plotting交互功能，查看时序光谱，以下是单季和双季水稻的时序曲线，双季呈现出双峰态，由于6月份数据受云影响较大，有效数据较少导致第一个峰略窄，单季是4-8月间的单峰态。<p>


```python
# 显示折线图
def show_broken_line(data,label,mode):
    data = np.squeeze(np.array(data))
    x = np.arange(len(data))
    plt.style.use('ggplot')
    plt.figure(figsize=(12,5))
    plt.plot(x, data,linestyle=":",color='darkviolet',linewidth = '2' )#, label="1", linestyle=":")
    plt.xticks(x,labels=label,rotation=60)
    plt.title(mode+"cropping")
    plt.ylabel("NDVI")
    plt.show()

def timeline_plot(x,y,mode):
    point=ee.Geometry.Point([x,y])
    timeline = geemap.extract_pixel_values(singleimg, point)
    name=list(timeline.keys().getInfo())
    values=np.array(timeline.values().getInfo())
    show_broken_line(values,[t for t in name],mode)
    return name,values
x1,y1=(116.40917424,28.68150022)
x2,y2=(116.27280998,28.71303109)
timeline_plot(x1,y1,'Single-')
timeline_plot(x2,y2,'Multiple-')
```

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/12.png)
![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/11.png)

   


### 3. 样本构建及机器学习推理

<p align="justify ">利用geemap的矢量勾绘功能结合时序光谱查看工具，对照前面的假彩色影像来圈定采样区域，确定一类保存一类，本案例圈定单季稻、双季稻以及其他地物三类。<p>


```python
## 获取矢量勾绘范围，可以构建多个多边形，用Map.user_rois进行抓取
# sampleregion1 = Map.user_rois
# sampleregion2 = Map.user_rois
# sampleregion3 = Map.user_rois
## 案例提供圈定的矢量文件，公众号末尾关键词
sampleregion1 = geemap.shp_to_ee('./data/shp/polygon_1.shp')
sampleregion2 = geemap.shp_to_ee('./data/shp/polygon_2.shp')
sampleregion0 = geemap.shp_to_ee('./data/shp/polygon_0.shp')
Map.addLayer(sampleregion1, {"color":'red'}, 'sampleregion1')
Map.addLayer(sampleregion2, {"color":'blue'}, 'sampleregion2')
Map.addLayer(sampleregion0, {"color":'green'}, 'sampleregion0')
# Map
```


```python
## 将不同label值作为属性添加到样本点中
def setLabel0(point):
  return point.set('label', 0) 
def setLabel1(point):
  return point.set('label', 1) 
def setLabel2(point):
  return point.set('label', 2) 
```


```python
points1 = ee.FeatureCollection.randomPoints(sampleregion1, 500).map(setLabel1)
points2 = ee.FeatureCollection.randomPoints(sampleregion2, 500).map(setLabel2)
points0 = ee.FeatureCollection.randomPoints(sampleregion0, 500).map(setLabel0)
Map.addLayer(points1, {"color":'red'}, 'sample1')
Map.addLayer(points2, {"color":'blue'}, 'sample2')
Map.addLayer(points0, {"color":'green'}, 'sample3')
Map
```


    Map(bottom=55039.0, center=[28.414352008722247, 836.5219071911866], controls=(WidgetControl(options=['position…

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/4.JPG)

构建样本训练集和分类器，并对整景影像做推理


```python
points=ee.FeatureCollection([points1,points2,points0]).flatten()
training = singleimg.sampleRegions(collection= points, properties= ['label'],scale= 10)
classifier = ee.Classifier.smileRandomForest(50).train(training, 'label', singleimg.bandNames())
classified = singleimg.select(bandname).classify(classifier)
```


```python
classVis = {'min': 0, 'max': 2,'palette': ['#FFFACD','#3CB371', '#4169E1']}
Map.addLayer(classified, classVis, 'prediction')
# Map
```


![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/7.JPG)

```python
singleimg=singleimg.setDefaultProjection('epsg:4326',None,10)
```


```python
# 这里提供了直接的下载方式和分波段的下载方式，后者下载过程中出错概率更小，数据量越大下载越不稳定，主要受网络和梯子的影响
geemap.download_ee_image(singleimg, filename='./data/ndvi_combine.tif',scale=10,region=region,crs='EPSG:4326')
# geemap.download_ee_image(classified, filename='./data/prediction.tif',scale=10,region=region,crs='EPSG:4326')

## 分波段下载在合并相对稳定
# for i in range(19,len(bandname.getInfo())):
#     img=output.select(i).clip(region.geometry())
#     outname = './data/ndvitimeline_'+str(i)+".tif"
#     geemap.download_ee_image(img, filename=outname,scale=10,region=region.geometry(),crs='EPSG:4326')
```

----------------------------------------
想了解更多请关注[45度科研人]公众号，欢迎给我留言！
<span style="display: block; text-align: center; margin-left: auto; margin-right: auto;">
    <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/wechat-simple.png" width="300"  alt="">
</span>
