---
title: "Segment-RS一个基于SAM的遥感智能交互解译工具"
author: 
  - "于峻川"
date: "2023-4-19"
categories:
  - Posts
  - Deep leanring
  - APP
image: "welcome.jpg"
toc: true
---

# Segment-RS一个基于SAM的遥感智能交互解译工具

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/1.png)


4月6号，facebook发布一种新的语义分割模型，Segment Anything Model (SAM)。仅仅3天时间该项目在Github就收获了1.8万个star,火爆程度可见一斑。有人甚至称之为CV领域的GPT时刻。我们也第一时刻对SAM模型进行了复现并用不同场景的遥感数据进行了测试，详见这篇文章：[第一个通用语义分割模型？Segment Anything 在遥感数据上的应用测试](https://junchuanyu.netlify.app/posts/2024-4-9seganything/)。

文章中我们也提到目前SAM更可能作为一种基础模型在细分领域迭代，最近一周已经有不少新的二创模型发布，在遥感领我们认为智能交互解译这种方式是AI在遥感方面拓展应用的重要媒介，而SAM中引入Prompt机制带来了更多的可能性。

目前基于SAM开展应用还有两方面问题需要解决，一个是需要一个便于交互的操作的界面，二是从算法层面解决SAM实例分割结果的分类问题。

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/insert.jpg)

最近我们开发了一个Segment-RS工具用来解决第一个问题，本地应用体验不错，在CPU上也能体验无延迟交互。我们在HF网站也部署了一个测试版本[Segment-RS](https://huggingface.co/spaces/JunchuanYu/SegRS
)，由于免费的CPU服务器配置有限，卡顿比较严重。针对第二个问题，我们也制定了初步的开发计划，感兴趣的朋友关注公众号动态。另外，上一篇文章发布后，有部分朋友表示不想复现SAM，但想看看SAM实际操作过程。于是我们基于新版的Segment-RS录制了一个操作视频。


<iframe width=800 height=500 src="https://player.bilibili.com/player.html?cid=1101334702&aid=867742543&page=1&as_wide=1&high_quality=1&danmaku=0" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

---------------------------
请关注微信公众号【45度科研人】获取更多精彩内容，欢迎后台留言！

<span style="display: block; text-align: center; margin-left: auto; margin-right: auto;">
    <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/wechat-simple.png" width="300"  alt="">
</span>
