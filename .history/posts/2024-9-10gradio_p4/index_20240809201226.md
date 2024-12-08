---
title: 使用Gradio构建交互式Web应用-P3
subtitle: Gradio与遥感数据处理（下）
author: 
  - "于峻川"
date: "2024-8-9"
categories:
  - Posts
  - Gradio
  - APP
  - Deep learning
image: https://dunazo.oss-cn-beijing.aliyuncs.com/blog/Gradio.png
toc: true
---


# 使用Gradio构建交互式Web应用

<br><br>
这是一个关于如何使用 Gradio 构建 Web 应用程序的开源系列教程。你将从设置 Python 环境开始，学习文本、图像等各类输入组件，自定义界面，设计复杂的交互等。本课程还将涵盖使用 Gradio 和 GDAL 处理遥感数据，用于图像增强、地理坐标转换、坡度分析等任务；学习如何使用 Gradio 和 Foliumap 创建交互式地图，实现动态地理空间数据可视化；如何集成机器学习模型并在 Hugging Face Spaces 上发布 web 应用程序。本教程包括实例、演示和作业。完成本教程后，你将能够高效地构建、部署和共享交互式 Web 应用程序。
课程相关配套请在文末获取。

<br><br>
课程相关资源链接[GITHUB](https://github.com/JunchuanYu/Building_Interactive_Web_APP_with_Gradio)

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/Gradio_07_2.png)

<br><br>

## Part3 ：Gradio与遥感数据处理（下）

<br><br>

### EMO 3-3: 地理空间坐标转换和shp矢量生成

<br><br>

本案例中将展示如何实现“度分秒”与“度”格式的经纬度进行转换，并且根据经纬度生成shp矢量，并下载到本地。构建该应用需要注意以下两点：

-  利用gepandas生成shp的时候，需要四个坐标点，这四个点是以左上为启示，逆时针依次循环的至右上。

- 生成的shp有多个文件构成，因此要使用zipfile对多个文件进行打包。下载打包后的文件需要利用gr.DownloadButton函数，注意它在generate_shp函数中是作为输入组件使用，而image_button使用的gr.DownloadButton是作为输出组件使用。

<br><br>

```python
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from shapely.geometry import Polygon
from pyproj import CRS
import zipfile
import os
import gradio as gr

# 定义起点和终点的十进制度数和度分秒格式
start_point =["109°35'57'',E", "41°50'49'',N"]  
end_point = ["110°12'24'',E","41°43'16'',N"]

start_point = ["109.57,E", "41.59,N"]  
end_point = ["110.12,E","41.16,N"]

# 将十进制度数转换为度、分和秒（DMS）格式的函数
def deg_to_dms(coord):
    deg, direction = coord.split(',')
    deg = float(deg)
    sign = 1 if direction in ['N', 'E'] else -1
    deg_abs = abs(deg)
    deg_int = int(deg_abs)*sign
    minutes = (deg_abs - deg_int) * 60
    seconds = (minutes - int(minutes)) * 60
    return f"{deg_int}° {int(minutes)}' {int(seconds)}'',{direction}",deg

# 将度、分和秒（DMS）格式转换为十进制度数的函数
def dms_to_deg(coord):
    parts = coord.replace('°', "'").replace("''", "'").replace(",", "'").split("'")
    deg = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    direction = parts[-1]
    decimal_deg = deg + (minutes / 60) + (seconds / 3600)
    sign = 1 if direction in ['N', 'E'] else -1
    decimal_deg = decimal_deg*sign
    return f"{decimal_deg:.4f},{direction}",decimal_deg

# 坐标转换器
def coord_convert(start_pointx,start_pointy,end_pointx,end_pointy,dms='True'):
    # print(dms)
    if dms=='True':
        textlon_min,lon_min=dms_to_deg(start_pointx)
        textlat_max,lat_max=dms_to_deg(start_pointy)
        textlon_max,lon_max=dms_to_deg(end_pointx)
        textlat_min,lat_min=dms_to_deg(end_pointy) 
    else:
        textlon_min,lon_min=deg_to_dms(start_pointx)
        textlat_max,lat_max=deg_to_dms(start_pointy)
        textlon_max,lon_max=deg_to_dms(end_pointx)
        textlat_min,lat_min=deg_to_dms(end_pointy)
    start_point=lon_min,lat_max
    end_point=lon_max,lat_min
    textstart_point=textlon_min,textlat_min
    textend_point=textlon_max,textlat_max
    return  textstart_point,textend_point,start_point,end_point

# 将Shapefile压缩成.zip文件以便下载的函数
def shp_to_zip(shp_file):
    directory = os.path.dirname(shp_file)
    basename = os.path.splitext(os.path.basename(shp_file))[0]
    extensions = ['.dbf', '.shx', '.cpg', '.prj','.shp']
    zip_filename = os.path.join(directory, f"{basename}.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for ext in extensions:
            filename = os.path.join(directory, f"{basename}{ext}")
            if os.path.exists(filename):
                zipf.write(filename, arcname=f"{basename}{ext}")
    # print(zip_filename)
    return zip_filename

# 从转换后的坐标生成Shapefile的函数
def generate_shp(start_point,end_point):
    print(start_point)
    print(end_point)
    vertices = [(start_point[0], start_point[1]), (start_point[0], end_point[1]),(end_point[0], end_point[1]),(end_point[0], start_point[1])]
    polygon = Polygon(vertices)
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=CRS('EPSG:4326').to_wkt())
    shpfile='./result.shp'
    gdf.to_file(shpfile)
    tempfile=shp_to_zip(shpfile)
    return gr.DownloadButton(label=f"Download {Path(tempfile).stem}", value=tempfile, visible=True)

with gr.Blocks(theme='NoCrypt/Miku') as demo:
    start_point=gr.State(None) 
    end_point=gr.State(None) 
    type=gr.State('False')
    with gr.Tab("Convert dms coord to shp"): # 度分秒格式坐标生成 shp
        with gr.Row():
            start_pointx=gr.Textbox(value="115°25'00'',E",label="Longtitude (top left)",interactive=True)
            start_pointy=gr.Textbox(value="41°03'00'',N",label="Latitude (top left)",interactive=True)
            end_pointx=gr.Textbox(value="117°30'00'',E",label="Longtitude (bottom right)",interactive=True)
            end_pointy=gr.Textbox(value="39°26'00'',N",label="Latitude(bottom right)",interactive=True)
        with gr.Row():
            convert_start=gr.Textbox(placeholder='',label="converted top left point")
            convert_end=gr.Textbox(placeholder='',label="converted bottom right point")
        with gr.Row():
            read_button=gr.Button("Read the corrd",visible=True,variant='primary')
            runbutton=gr.Button("Generate SHP",variant='primary')
            image_button=gr.DownloadButton("Download",visible=True,variant='secondary')
                    
        read_button.click(coord_convert,[start_pointx,start_pointy,end_pointx,end_pointy],[convert_start,convert_end,start_point,end_point])
        runbutton.click(generate_shp, [start_point,end_point],image_button)   
        
    with gr.Tab("Convert degree coord to shp"): # 以度为单位的坐标生成shp
        with gr.Row():
            start_pointx=gr.Textbox(value="115.42,E",label="Longtitude (top left)",interactive=True)
            start_pointy=gr.Textbox(value="41.05,N",label="Latitude (top left)",interactive=True)
            end_pointx=gr.Textbox(value="117.50,E",label="Longtitude (bottom right)",interactive=True)
            end_pointy=gr.Textbox(value="39.43,N",label="Latitude(bottom right)",interactive=True)
        with gr.Row():
            convert_start=gr.Textbox(placeholder='',label="converted top left point")
            convert_end=gr.Textbox(placeholder='',label="converted bottom right point")
        with gr.Row():
            read_button=gr.Button("Read the corrd",visible=True,variant='primary')
            runbutton=gr.Button("Generate SHP",variant='primary')
            image_button=gr.DownloadButton("Download",visible=True,variant='secondary')
            
        read_button.click(coord_convert,[start_pointx,start_pointy,end_pointx,end_pointy,type],[convert_start,convert_end,start_point,end_point])
        runbutton.click(generate_shp, [start_point,end_point],image_button)  

```

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/demo3-3_new.gif)



---------------------------
请关注微信公众号【45度科研人】回复“**@gradio**”获取该教程配套数据，欢迎后台留言！


<span style="display: block; text-align: center; margin-left: auto; margin-right: auto;">
    <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/wechat-simple.png" width="200"  alt="">
</span>

---------------------------

为了促进沟通与交流，我们建立了「养生科研」学术交流群。这个平台不仅能够让大家迅速获取本公众号的资源，还为各位提供了一个共同探讨问题、交流思想的空间。有意向加入交流群的朋友们，可以通过添加小编的微信来获得入群邀请。请注意，在添加时请按照“加群-单位-研究方向-姓名”的格式备注您的信息，否则您的申请可能无法通过。


<span style="display: block; text-align: center; margin-left: auto; margin-right: auto;">
    <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/laidian.jpg" width="200"  alt="">
</span>
