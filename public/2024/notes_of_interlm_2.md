## 第2课4/3


https://github.com/InternLM/Tutorial/blob/camp2/helloworld/hello_world.md

> 总结：图片理解能力不错，图文生成速度略慢

#### 1、部署1.8B模型
- 开发机：10%GPU，cuda11.7
- terminal环境配置
```bash
studio-conda -o internlm-base -t demo
# 与 studio-conda 等效的配置方案
# conda create -n demo python==3.10 -y
# conda activate demo
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

## 进入conda环境
conda activate demo

## 安装依赖包
pip install huggingface-hub==0.17.3  transformers==4.34 psutil==5.9.8 accelerate==0.24.1 streamlit==1.32.2 matplotlib==3.8.3 modelscope==1.9.5 sentencepiece==0.1.99

===================
### 2 下载 `InternLM2-Chat-1.8B` 模型
===================
mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo

## 打开 `/root/demo/download_mini.py` 文件，复制代码

import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"
snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')

===================
## 执行命令，下载模型参数文件：
python /root/demo/download_mini.py

===================
### 3 运行 cli_demo
===================
## 打开 `/root/demo/cli_demo.py` 文件，复制

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

===================
## 输入命令，执行 Demo 程序：
conda activate demo
python /root/demo/cli_demo.py

## 等待模型加载完成，键入内容示例：
User  >>> 请创作一个 300 字、有关小猫咪的小故事

在一个美丽的花园里，住着一只名叫小咪的小猫咪。小咪非常喜欢在花园里玩耍，尤其是当它看到那些美丽的花朵时，总是忍不住地停下来，用它的小爪子轻轻地抚摸花瓣，仿佛在和它们进行一场心灵的交流。

有一天，小咪发现花园里有一只小兔子在玩耍。它好奇地看着小兔子，然后悄悄地走近，想看看小兔子到底在玩什么游戏。小兔子看到小咪来了，它停下了玩耍，转过头来看着小咪。

小咪开始轻轻地舔小兔子的毛发，就像是在和它说话一样。小兔子被小咪的举动所感动，它开始和小咪玩耍起来，一边唱歌，一边跳着舞。小咪和小兔子成为了好朋友，它们一起在花园里玩耍，享受着美好的时光。

小咪发现，只要和动物交朋友，就可以享受到它们所带来的快乐和温暖。它决定继续和花园里的其他动物交朋友，让它的小世界变得更加美好和充满生机。

```

### 3 **实战：部署 `八戒-Chat-1.8B` 模型**
- 介绍：https://www.modelscope.cn/models/JimmyMa99/BaJie-Chat-mini/summary
```bash
===================
### 2 **配置基础环境**
===================
conda activate demo

## 使用 `git` 命令来获得仓库内的 Demo 文件
cd /root/
git clone https://gitee.com/InternLM/Tutorial -b camp2
# git clone https://github.com/InternLM/Tutorial -b camp2
cd /root/Tutorial

===================
### 3 **下载运行 Chat-八戒 Demo**
===================
## 在 `Web IDE` 中执行 `bajie_download.py`
python /root/Tutorial/helloworld/bajie_download.py

## 下载完成后，输入运行命令
streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006

## 程序运行的同时，对端口环境配置本地 `PowerShell` （**Mac 用户打开终端即可**）
## 打开 PowerShell 后，先查询端口，再根据端口键入命令 （例如图中端口示例为 38374）
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 40773

## 打开(http://127.0.0.1:6006/) 后，等待加载完成即可进行对话
》〉》你觉得人间怎么样
嘿嘿，人间啊，俺老猪觉得人间可真是热闹非凡的地方！有吃有喝，还有美女，简直是俺老猪的天堂啊！不过，俺老猪可不喜欢那些妖精和妖怪，总是想要吃掉人间的美食，真是让人头疼。而且人间的规矩可多了，有些规矩俺老猪是不懂的，比如那些什么礼仪礼貌，俺老猪就觉得有点太繁琐了。不过，人间的风景也挺好的，有山有水，还有各种各样的花，俺老猪就喜欢看着这些美景，心情也变得愉快起来。总之，人间嘛，有好吃的、有好玩的，就是有些规矩和繁琐的地方让俺老猪不太适应，不过，还是愿意留在人间享受这些美食和美景的！

```

### 4 **实战：使用 `Lagent` 运行 `InternLM2-Chat-7B` 模型（开启 30% A100 )**
[![alt text](https://github.com/InternLM/Tutorial/raw/camp2/helloworld/images/Lagent-1.png)](https://github.com/InternLM/Tutorial/blob/camp2/helloworld/images/Lagent-1.png)
**Lagent 的特性**总结如下：
- 流式输出：提供 stream_chat 接口作流式输出，本地就能演示酷炫的流式 Demo。
- **接口统一，设计全面升级，提升拓展性**，包括：
    - **Model** : 不论是 OpenAI API, Transformers 还是推理加速框架 LMDeploy 一网打尽，模型切换可以游刃有余；
    - **Action**: 简单的继承和装饰，即可打造自己个人的工具集，不论 InternLM 还是 GPT 均可适配；
    - **Agent**：与 Model 的输入接口保持一致，模型到智能体的蜕变只需一步，便捷各种 agent 的探索实现；
- 文档全面升级，API 文档全覆盖。
- 关闭开发机，【升降】配置
- 附录：小技巧
```bash
===================
### 2 **配置基础环境**
===================
conda activate demo

cd /root/demo

## 使用 git 命令下载 Lagent 相关的代码库
git clone https://gitee.com/internlm/lagent.git
# git clone https://github.com/internlm/lagent.git
cd /root/demo/lagent
git checkout 581d9fb8987a5d9b72bb9ebd37a95efd47d479ac
pip install -e . # 源码安装

===================
### 3 **使用 `Lagent` 运行 `InternLM2-Chat-7B` 模型为内核的智能体**
===================
cd /root/demo/lagent

## 构造软链接快捷访问方式：
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b

## 打开 `lagent` 路径下 `examples/internlm2_agent_web_demo_hf.py` 文件，并修改对应位置 (71行左右) 代码
value='/root/models/internlm2-chat-7b'

## 输入运行命令 - **点开 6006 链接后，大约需要 5 分钟完成模型加载**
streamlit run /root/demo/lagent/examples/internlm2_agent_web_demo_hf.py --server.address 127.0.0.1 --server.port 6006

## 待程序运行的同时，对本地端口环境配置本地 `PowerShell` 。使用快捷键组合 `Windows + R`（Windows 即开始菜单键）打开指令界面，并输入命令，按下回车键。（Mac 用户打开终端即可）打开 PowerShell 后，先查询端口，再根据端口键入命令

# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 40986

## 打开 [http://127.0.0.1:6006]后，（会有较长的加载时间）勾上数据分析，其他的选项不要选择，进行计算方面的 Demo 对话，即完成本章节实战
请解方程 2*X*X=1360 之中 X 的结果

==============
数据分析提示词
==============
你现在已经能够在一个有状态的 Jupyter 笔记本环境中运行 Python 代码。当你向 python 发送含有 Python 代码的消息时，它将在该环境中执行。这个工具适用于多种场景，如数据分析或处理（包括数据操作、统计分析、图表绘制），复杂的计算问题（解决数学和物理难题），编程示例（理解编程概念或特性），文本处理和分析（比如文本解析和自然语言处理），机器学习和数据科学（用于展示模型训练和数据可视化），以及文件操作和数据导入（处理CSV、JSON等格式的文件）。

```
## 5 **实战：实践部署 `浦语·灵笔2` 模型（开启 50% A100）**
### 5.1 **初步介绍 `XComposer2` 相关知识**
`浦语·灵笔2` 是基于 `书生·浦语2` 大语言模型研发的突破性的图文多模态大模型，具有非凡的图文写作和图像理解能力，在多种应用场景表现出色，总结起来其具有：
- 自由指令输入的图文写作能力： `浦语·灵笔2` 可以理解自由形式的图文指令输入，包括大纲、文章细节要求、参考图片等，为用户打造图文并貌的专属文章。生成的文章文采斐然，图文相得益彰，提供沉浸式的阅读体验。
- 准确的图文问题解答能力：`浦语·灵笔2` 具有海量图文知识，可以准确的回复各种图文问答难题，在识别、感知、细节描述、视觉推理等能力上表现惊人。
- 杰出的综合能力： `浦语·灵笔2-7B` 基于 `书生·浦语2-7B` 模型，在13项多模态评测中大幅领先同量级多模态模型，在其中6项评测中超过 `GPT-4V` 和 `Gemini Pro`。

```shell
conda activate demo
# 补充环境包
pip install timm==0.4.12 sentencepiece==0.1.99 markdown2==2.4.10 xlsxwriter==3.1.2 gradio==4.13.0 modelscope==1.9.5

===================
下载 **InternLM-XComposer 仓库** 相关的代码资源
===================
cd /root/demo
git clone https://gitee.com/internlm/InternLM-XComposer.git
# git clone https://github.com/internlm/InternLM-XComposer.git
cd /root/demo/InternLM-XComposer
git checkout f31220eddca2cf6246ee2ddf8e375a40457ff626

===================
构造软链接快捷访问方式
===================
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b /root/models/internlm-xcomposer2-7b
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b /root/models/internlm-xcomposer2-vl-7b

===================
### 3 **图文写作实战
===================
# 继续输入指令，用于启动 `InternLM-XComposer`
python /root/demo/InternLM-XComposer/examples/gradio_demo_composition.py  \
--code_path /root/models/internlm-xcomposer2-7b \
--private \
--num_gpus 1 \
--port 6006

# 待程序运行的同时，参考章节 3.3 部分对端口环境配置本地 `PowerShell`
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 40986

根据以下标题：“中国水墨画：流动的诗意与东方美学”，创作长文章，字数不少于800字。请结合以下文本素材：
“水墨画是由水和墨调配成不同深浅的墨色所画出的画，是绘画的一种形式，更多时候，水墨画被视为中国传统绘画，也就是国画的代表。也称国画，中国画。墨水画是中国传统画之一。墨水是国画的起源，以笔墨运用的技法基础画成墨水画。线条中锋笔，侧锋笔，顺锋和逆锋，点染，擦，破墨，拨墨的技法。墨于水的变化分为五色。画成作品，题款，盖章。就是完整的墨水画作品。
基本的水墨画，仅有水与墨，黑与白色，但进阶的水墨画，也有工笔花鸟画，色彩缤纷。后者有时也称为彩墨画。在中国画中，以中国画特有的材料之一，墨为主要原料加以清水的多少引为浓墨、淡墨、干墨、湿墨、焦墨等，画出不同浓淡（黑、白、灰）层次。别有一番韵味称为“墨韵”。而形成水墨为主的一种绘画形式。”

===================
### 4 **图片理解实战**
===================
# 关闭并重新启动一个新的 `terminal`，继续输入指令，启动 `InternLM-XComposer2-vl`
conda activate demo

cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_chat.py  \
--code_path /root/models/internlm-xcomposer2-vl-7b \
--private \
--num_gpus 1 \
--port 6006

请分析一下图中内容


```
![[WechatIMG785.jpg]]



