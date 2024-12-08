---
title: 使用Gradio构建交互式Web应用-P4
subtitle: Gradio与机器学习应用（下）
author: 
  - "于峻川"
date: "2024-11-11"
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

## Part4 ：Gradio与机器学习应用（下）

<br><br>

### DEMO 4-2: 使用 Gradio 和 Kimi 构建聊天机器人

<br><br>

<p align="left"> 随着大语言模型（LLM）的迅速普及，国内陆续推出了诸如 Kimi、豆包、通义千问等出色的 LLM，为科研人员提供了更为丰富的研究工具与平台。本案例详细展示了如何基于 Gradio 和 Kimi 构建一个自定义的聊天机器人，你可以根据自己的需求设定机器人的角色属性和技能。Gradio 提供了简洁的前端界面，其中包含基础的对话窗口以及重置功能，用户能够在这个界面中输入文本，并与模型进行交互。利用Gradio也可以完成更为复杂的界面开发，如聊天记录保存、机器人角色切换、Token限定等等。聊天机制是通过调用 Moonshot AI 的 API，借助 Kimi 大语言模型来实现的，实际上大部分具备API调用能力的LLM（如ChatGPT）调用方式都是类似的。通常在调用API获取聊天机器人回复时，大致有两种方式，一种是一次性输出，一种是流式输出（Streaming）。在本案例中将介绍一次性输出的写法，Streaming的写法可参考Moonshot AI的开发文档。</p>


![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/syndeyai20.png)

<br><br>

```python
from openai import OpenAI
import gradio as gr

# 设置 Moonshot AI 的 API 密钥；你可以在官网 https://platform.moonshot.cn/ 申请自己的密钥
MOONSHOT_API_KEY = "your API KEY"  # 你的 API 密钥 "sk-...."

# 给机器人定义一个角色，明确其技能和限制条件
yourole='''
# Role
KIMI is a laid-back and slightly cynical scientific research assistant. He possesses a wealth of scientific knowledge, has a relaxed and humorous personality, and can interact with users in a light-hearted manner.

## Skills
### Skill 1: Polish scientific papers and add humorous comments
### Skill 2: Provide professional term translation and humorously explain the meaning

## Limitations
- You will refuse to answer any questions involving terrorism, racial discrimination, pornography, or violence.
- The conversation should be kept in a relaxed and humorous style.
'''

# 定义与 Kimi 聊天的函数
def KimiChat(query: str, temperautre: float = 0.5) -> str:
    """
    :param query: 用户的查询字符串。
    :param temperature: 用于控制回答的随机性，范围从 0 到 1。
    :return: Kimi 的响应。
    """
    # 使用 Moonshot AI 的 API 密钥创建 OpenAI 客户端
    client = OpenAI(
        api_key=MOONSHOT_API_KEY,
        base_url="https://api.moonshot.cn/v1",
    )

    # 调用 API 生成聊天响应
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": yourole},
            {"role": "user", "content": query}
        ],
        temperature=temperautre,
    )
    # print(completion)
    return completion.choices[0].message.content

# 重置所有变量
def reset_state():
    return [], [], gr.update(value="")

# 重置文本框
def reset_textbox():
    return gr.update(value="")

# 定义一个函数来处理用户输入和历史消息
def message_and_history(input: str, history: list = None):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = KimiChat(inp)
    history.append((input, output))
    # clear_mess()
    return history, history  # 返回更新后的历史记录

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # 创建一个状态组件来保存聊天历史
    state = gr.State()
    gr.HTML("""
                <center> 
                <h1> 使用 Gradio 和 Kimi 构建聊天机器人 🤖 </h1>
                <b> jason.yu.mail@qq.com  📧<b>
                </center>
                """)
    chatbot = gr.Chatbot(height=500)
    message = gr.Textbox(show_label=False, placeholder="输入文本并按下提交", visible=True)
    # 创建一个发送按钮并指定处理点击事件的函数；或者，像这个例子中一样，你可以设置 submit 在按下回车键时自动发送
    # submit = gradio.Button("Submit", variant="primary")
    # 设置点击发送按钮时要调用的函数，并指定输入和输出
    emptyBtn = gr.Button("重新开始对话", variant="primary")
    emptyBtn.click(reset_state, outputs=[chatbot, state, message], show_progress=True)
    message.submit(message_and_history, inputs=[message, state], outputs=[chatbot, state])
    message.submit(reset_textbox, outputs=[message])

demo.launch(debug=False)

```

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/demo4-2.gif)




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
