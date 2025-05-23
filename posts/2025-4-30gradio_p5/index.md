---
title: 使用Gradio构建交互式Web应用-P5
subtitle: 使用Gradio构建交互式地图应用
author: 
  - "于峻川"
date: "2025-4-30"
categories:
  - Posts
  - Gradio
  - APP
  - Deep learning
image: https://dunazo.oss-cn-beijing.aliyuncs.com/blog/Gradio_11.png
toc: true
---

# 使用Gradio构建交互式Web应用

<br><br>
这是一个关于如何使用 Gradio 构建 Web 应用程序的开源系列教程。你将从设置 Python 环境开始，学习文本、图像等各类输入组件，自定义界面，设计复杂的交互等。本课程还将涵盖使用 Gradio 和 GDAL 处理遥感数据，用于图像增强、地理坐标转换、坡度分析等任务；学习如何使用 Gradio 和 Foliumap 创建交互式地图，实现动态地理空间数据可视化；如何集成机器学习模型并在 Hugging Face Spaces 上发布 web 应用程序。本教程包括实例、演示和作业。完成本教程后，你将能够高效地构建、部署和共享交互式 Web 应用程序。
课程相关配套请在文末获取。

<br><br>

## Part5 ：使用Gradio构建交互式地图应用

<br><br>

### DEMO 5-1: 交互地图框架构建与坐标定位

<br><br>

将坐标范围或影像显示在地图中是遥感处理的常规操作。Gradio中虽然没有直接提供相应的控件，但可以借助Leafmap中的Folium等实现交互式地图的构建.本案例中主要呈现含有在线影像的底图框架的构建以及坐标的定位。

- **Leafmap** (leafmap.org) 是一个专为交互式地理空间分析和可视化设计的Python库，由吴秋生老师创建。它整合了多个流行工具（如 Folium、ipyleaflet 和 Google Earth Engine），简化了地理数据处理和地图创建的流程。
- **地图框架** 能够实现在Gradio中呈现动态地图的关键在于，通过leafmap的to_gradio函数将地图转为Gradio可接收的Html格式，从而实现地图框架的构建。

<br><br>

```python
import gradio as gr
import leafmap.foliumap as leafmap

# 定义一个根据参数生成地图的函数
def generate_map(zoom_level, maptype="Esri.WorldStreetMap", coordsy='', coordsx=''):
    if coordsy == '' and coordsx == '':
        coordsy = 40   # 默认纬度坐标
        coordsx = 116.3  # 默认经度坐标
    print(maptype)

    # 使用leafmap创建地图对象，指定中心点和缩放级别
    map = leafmap.Map(location=(coordsy, coordsx), zoom=zoom_level)
    map.add_basemap(maptype)  # 添加指定的底图类型
    return map.to_gradio()    # 返回HTML格式的地图

# 创建Gradio界面
with gr.Blocks() as demo:
    # 顶部标题
    gr.HTML("""
            <center> 
            <h1> General a map 🗺️ </h1>
            <b> jason.yu.mail@qq.com  📧<b>
            </center>
            """)      
    with gr.Row():
      with gr.Row():
        # 经纬度输入框
        coordinates_input_y = gr.Textbox(value='',placeholder=40,label="中心点纬度",lines=1)
        coordinates_input_x = gr.Textbox(value='',placeholder=116.3,label="中心点经度",lines=1)
        # 缩放级别滑块
        zoom_level_input = gr.Slider(value=9,minimum=4,maximum=15,step=1,label="选择缩放级别",interactive=True)

    with gr.Row():
      # 底图类型下拉菜单
      maptype=gr.Dropdown(
              choices=[
                  "Esri.NatGeoWorldMap",
                  "Esri.WorldGrayCanvas",
                  "Esri.WorldImagery",
                  "Esri.WorldShadedRelief",
                  "Esri.WorldStreetMap",
                  "Esri.WorldTerrain",
                  "Esri.WorldTopoMap",
              ],value="Esri.WorldStreetMap",interactive=True,label="底图类型")
      # 生成地图按钮
      map_button = gr.Button("生成地图",scale=1)
    with gr.Row():
      # 地图输出区域
      map_output = gr.HTML() 

    # 按钮点击事件绑定
    map_button.click(generate_map, inputs=[zoom_level_input,maptype,coordinates_input_y,coordinates_input_x], outputs=[map_output])

# 启动多线程处理模式
demo.queue().launch() 

# infile='iputfile'
# path='https://github.com/JunchuanYu/Gradio_tutorial/blob/main/data/raster.tif'

```

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/demo5-1.gif)




### DEMO 5-2: Tiff文件的云端可视化

<br><br>

栅格数据的类型是多样的，由于Gradio构建的交互式应用是依托Web构建的，因此Tiff需要转为栅格切片服务才能正确的呈现在地图中。本案例中以保存在项目目录下的“input.tif”和保存在网络的“https://github.com/JunchuanYu/Gradio_tutorial/blob/main/data/raster.tif”的可视化为例进行说明。

- **在线底图** Folium中支持在线地图瓦片的加载，除了esri、arcgis、google之外也支持天地图等国产数据。
- **本地Tiff** 其可视化过程是将Tiff转为本地的瓦片图层再叠加到底图中。
- **云端Tiff** 本案例中的云端Tiff是先下载到本地后再进行可视化的，此外，可以采用add_remote_tile 函数用于加载 Cloud Optimized GeoTIFF (COG) 格式的远程 TIFF 文件直接加载。

<br><br>

```python
import gradio as gr
import leafmap.foliumap as leafmap

# 定义一个函数，用于在leafmap中加载Tiff文件
def showtiff(text1, text2):
    infile = str(text1)  # 转换为字符串格式
    filepath = str(text2)  # 转换为字符串格式
    
    # 创建地图对象
    Map = leafmap.Map()
    
    # 添加默认底图
    Map.add_tile_layer(
        infile='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        name='arcgisonline',
        attribution='attribution'
    )
    
    # 如果infile不为空，则添加瓦片图层
    if infile != '':
        Map.add_raster(infile)
    
    # 如果文件路径不为空，则下载并添加Tiff文件
    if filepath != '':
        raster = "raster.tif"
        raster = leafmap.download_file(filepath, "raster.tif")
        Map.add_raster(raster, layer_name='insar')
    
    return Map.to_gradio()

# 创建Gradio界面
with gr.Blocks(theme='gradio/soft') as demo:
    # 页面标题
    gr.HTML("""
            <center> 
            <h1> 使用foliumap处理地图数据 🗺️ </h1>
            <b> jason.yu.mail@qq.com  📧<b>
            </center>
            """)
    
    # 输入行
    with gr.Row():
        input = gr.Textbox(label='输入本地Tiff文件名', interactive=True)
        input2 = gr.Textbox(label='输入Tiff文件URL', interactive=True)
    
    # 输出行
    with gr.Row():
        out = gr.HTML()
    
    # 绑定事件处理
    input.change(showtiff, inputs=[input, input2], outputs=out)
    input2.change(showtiff, inputs=[input, input2], outputs=out)

# 启动应用
demo.launch(debug=True)

```

<br><br>

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/newdemo5-2.gif)




---------------------------
请关注微信公众号【45度科研人】回复“**@gradio**”获取该教程配套数据，欢迎后台留言！


<span style="display: block; text-align: center; margin-left: auto; margin-right: auto;">
    <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/wechat-simple.png" width="200"  alt="">
</span>


