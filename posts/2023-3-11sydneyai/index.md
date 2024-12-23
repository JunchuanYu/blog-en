---
title: " Sydney-AI, a free ChatGPT platform "
author: 
  - "于峻川 (Junchuan Yu)"
date: "2023-3-11"
categories:
  - Posts
  - Deep leanring
  - APP
image: "welcome.jpg"
toc: true
---
![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/SYDNEYAI.jpg)

# Sydney-AI, a free ChatGPT platform
We have developed a ChatGPT web platform based on the latest OpenAI interface, ChatGPT-3.5-turbo, to provide free services to the friends of「45 度科研人」. After two weeks of testing, the platform is running stably. If you encounter any problems during use, please let us know. The operating costs of the platform are covered by the 「45 度科研人」 public account, and we appreciate your support!

Sydney-AI: [https://junchuanyu-sydney-ai.hf.space](https://junchuanyu-sydney-ai.hf.space)

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/6.JPG)

## How can researchers benefit from using Sydney-AI?
I believe everyone is familiar with ChatGPT by now. We think that ChatGPT is a great auxiliary software for researchers. It has impressive performance in data collection, language refinement, auxiliary development, and text editing. During the two weeks of in-depth contact with Sydney-AI, our team has become accustomed to working with ChatGPT to complete tasks. If you want to learn more about ChatGPT or New Bing, please refer to a previous article：[New Bing？也许是New + Everything！](https://junchuanyu.netlify.app/posts/2023-3-4newbing/).

## What are the unique features of Sydney-AI? Why was it given this name?
The convenience of using ChatGPT is the original intention of our application development. The features of Sydney-AI solve two pain points in using ChatGPT in China, network and cross-device issues.
 - Sydney-AI can be accessed without magic 
 - Sydney-AI can be used on both computer and mobile devices 
 - The name Sydney is what the new version of ChatGPT calls itself. During the initial testing stage of New Bing, ChatGPT showed rich emotions. It was humorous, talkative, and even flirtatious. Many people loved this non-robotic performance of Sydney. However, for various reasons, Microsoft has updated New Bing and the emotionally rich Sydney has disappeared. Therefore, naming it Sydney-AI expresses our expectation for the new open API interface. 

## How to use Sydney-AI？
### Basic settings：

 - There is no limit on the number of conversations, but the number of tokens per conversation is limited to 3000. If you exceed this value, you need to start a new conversation. Tokens are different from the number of words. 1000 tokens are roughly equivalent to 750 words. 
 - Sydney-AI is free to use and is already built-in with OpenAI's API Key. If you want to exceed the 3000-token limit, you can also fill in your own API Key in the API Key window. 
 - The software has some commonly used roles of ChatGPT built-in, which can be selected through a drop-down menu. The default is the original ChatGPT. You can also add role descriptions in the input text to make ChatGPT play the role you want. For example, "You are a university teacher in remote sensing, please popularize the concept of remote sensing to elementary school students in a simple and easy-to-understand way." Or "I hope you can be a role of English translation, spelling correction, and improvement. I will communicate with you in any language, and you will detect the language, translate it, and answer in the corrected and improved English." 

### Method of use
- You can enter your centence in the text box. 
- Two adjustable parameters are provided. When the Max tokens number is set to greater than 3000, you need to input your own OpenAI Key. It was found that 3000 tokens can satisfy most usage scenarios. When the Temperature value is set higher, the answer will be more divergent, but the probability of errors will also increase. 
- OpenAI API calling method. When you need ChatGPT to complete more difficult tasks, such as converting the reply text into a CSV file and downloading it to your local computer, you need to expand the API function according to your needs. The following is the calling method of the new version of the API, and it is recommended to use Colab for debugging.

```python
import openai
# enter your openai api key
openai.api_key = ”OPENAI_API_KEY“
# enter your prompt
prompt = ‘enter centence here’
response = openai.ChatCompletion.create(
    model=‘gpt-3.5-turbo ’,
    messages=[
        {”role“: ”system“, ”content“: ”you are a helpful assistant.“},
        {”role“: ”user“, ”content“: prompt}
    ]
)
# get the response
resText = response.choices[0].message.content
print(resText)
```


----------------------------------------
Kindly consider following the official WeChat account 「45 度科研人」 to access more interesting content. Feel free to leave a message in the background.

<span style="display: block; text-align: center; margin-left: auto; margin-right: auto;">
    <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/wechat-simple.png" width="300"  alt="">
</span>