---
title: 使用Gradio构建交互式Web应用-P4
subtitle: Gradio与机器学习应用（上）
author: 
  - "于峻川"
date: "2024-9-10"
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
<p align="left"> 这是一个关于如何使用 Gradio 构建 Web 应用程序的开源系列教程。你将从设置 Python 环境开始，学习文本、图像等各类输入组件，自定义界面，设计复杂的交互等。本课程还将涵盖使用 Gradio 和 GDAL 处理遥感数据，用于图像增强、地理坐标转换、坡度分析等任务；学习如何使用 Gradio 和 Foliumap 创建交互式地图，实现动态地理空间数据可视化；如何集成机器学习模型并在 Hugging Face Spaces 上发布 web 应用程序。本教程包括实例、演示和作业。完成本教程后，你将能够高效地构建、部署和共享交互式 Web 应用程序。</p>

<br><br>
课程相关资源链接[GITHUB](https://github.com/JunchuanYu/Building_Interactive_Web_APP_with_Gradio)

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/Gradio_09.png)


<br><br>

## Part4 ：Gradio与机器学习应用（上）

<br><br>

### DEMO 4-1: 手写数字识别APP

<br><br>

<p align="left"> 本案例展示了一个基于随机森林算法的手写字母识别系统。在手写字母识别任务中，虽然可以选择传统的机器学习算法如随机森林或支持向量机，也可以采用更先进的卷积神经网络，但在本案例中，为了演示目的，我们选择了训练成本相对较低的随机森林算法。通过Gradio库，我们创建了一个用户友好的前端界面，用户可以轻松地在画布上绘制字母，而模型会即时给出预测结果。Gradio因其易用性和灵活性，已成为展示深度学习算法的流行前端框架之一。</p>


<br><br>

```python
# 训练一个随机森林模型
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 
import numpy as np  

# 加载数据
path = './data/mnist.npz'  
with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"]  # 训练数据和标签
    x_test, y_test = f["x_test"], f["y_test"]  # 测试数据和标签

print(x_train.shape, y_train.shape)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
# 训练模型
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# 将训练好的模型保存到本地
joblib.dump(clf, './data/random_forest_model.pkl')

# 导入Gradio库，用于创建交互式应用
import gradio as gr
import joblib
import numpy as np

# 加载预先训练好的随机森林模型
model = joblib.load('./data/random_forest_model.pkl')

# 定义预测函数
def predict_minist(image):
    normalized = image['composite'][:, :, -1]
    flattened = normalized.reshape(1, 784)
    prediction = model.predict(flattened)
    print(normalized.shape, np.max(normalized), prediction[0])
    return prediction[0]

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""
        <center> 
        <h1>Handwritten Digit Recognition</h1>
        <b>jason.yu.mail@qq.com 📧</b>
        </center>
        """)  
    # 添加Markdown组件，提示用户在画布中心绘制数字
    gr.Markdown("Draw a digit and the model will predict the digit. Please draw the digit in the center of the canvas")
    with gr.Row():
        outtext = gr.Textbox(label="Prediction")
    with gr.Row():
        inputimg = gr.ImageMask(image_mode="RGBA", crop_size=(28,28))

    # 构建监听机制，当输入change时，对图像进行预测
    inputimg.change(predict_minist, inputimg, outtext)
# 定义demo的网页尺寸
demo.launch(height=550,width="100%",show_api=False)

```

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/demo4-1.gif)




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
