# VisualGLM-6B

<p align="center">
🤗 <a href="https://huggingface.co/THUDM/visualglm-6b" target="_blank">HF Repo</a> • ⚒️ <a href="https://github.com/THUDM/SwissArmyTransformer" target="_blank">SwissArmyTransformer (sat)</a> • 🐦 <a href="https://twitter.com/thukeg" target="_blank">Twitter</a> 
</p>
<p align="center">
•  📃 <a href="https://arxiv.org/abs/2105.13290" target="_blank">[CogView@NeurIPS 21]</a>  <a href="https://github.com/THUDM/CogView" target="_blank">[GitHub]</a> • 📃 <a href="https://arxiv.org/abs/2103.10360" target="_blank">[GLM@ACL 22]</a> <a href="https://github.com/THUDM/GLM" target="_blank">[GitHub]</a> <br>
</p>
<p align="center">
    👋 加入我们的 <a href="https://join.slack.com/t/chatglm/shared_invite/zt-1th2q5u69-7tURzFuOPanmuHy9hsZnKA" target="_blank">Slack</a> 和 <a href="examples/WECHAT.md" target="_blank">WeChat</a>
</p>
<!-- <p align="center">
🤖<a href="https://huggingface.co/spaces/THUDM/visualglm-6b" target="_blank">VisualGLM-6B在线演示网站</a>
</p> -->

## 介绍

VisualGLM-6B is an open-source, multi-modal dialog language model that supports **images, Chinese, and English**. The language model is based on [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) with 6.2 billion parameters; the image part builds a bridge between the visual model and the language model through the training of [BLIP2-Qformer](https://arxiv.org/abs/2301.12597), with the total model comprising 7.8 billion parameters. **[Click here for English version.](README_en.md)**

VisualGLM-6B 是一个开源的，支持**图像、中文和英文**的多模态对话语言模型，语言模型基于 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)，具有 62 亿参数；图像部分通过训练 [BLIP2-Qformer](https://arxiv.org/abs/2301.12597) 构建起视觉模型与语言模型的桥梁，整体模型共78亿参数。

VisualGLM-6B 依靠来自于 [CogView](https://arxiv.org/abs/2105.13290) 数据集的30M高质量中文图文对，与300M经过筛选的英文图文对进行预训练，中英文权重相同。该训练方式较好地将视觉信息对齐到ChatGLM的语义空间；之后的微调阶段，模型在长视觉问答数据上训练，以生成符合人类偏好的答案。

VisualGLM-6B 由 [SwissArmyTransformer](https://github.com/THUDM/SwissArmyTransformer)(简称`sat`) 库训练，这是一个支持Transformer灵活修改、训练的工具库，支持Lora、P-tuning等参数高效微调方法。本项目提供了符合用户习惯的huggingface接口，也提供了基于sat的接口。

结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4量化级别下最低只需8.7G显存）。

-----

VisualGLM-6B 开源模型旨在与开源社区一起推动大模型技术发展，恳请开发者和大家遵守开源协议，勿将该开源模型和代码及基于该开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。目前，本项目官方未基于 VisualGLM-6B 开发任何应用，包括网站、安卓App、苹果 iOS应用及 Windows App 等。

由于 VisualGLM-6B 仍处于v1版本，目前已知其具有相当多的[**局限性**](README.md#局限性)，如图像描述事实性/模型幻觉问题，图像细节信息捕捉不足，以及一些来自语言模型的局限性。尽管模型在训练的各个阶段都尽力确保数据的合规性和准确性，但由于 VisualGLM-6B 模型规模较小，且模型受概率随机性因素影响，无法保证输出内容的准确性，且模型易被误导（详见局限性部分）。在VisualGLM之后的版本中，将会着力对此类问题进行优化。本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。

## 样例
VisualGLM-6B 可以进行图像的描述的相关知识的问答。
![泰坦尼克号样例](examples/chat_example1.png)

<details>
<summary>也能结合常识或提出有趣的观点，点击展开/折叠更多样例</summary>

![出租车熨衣服样例](examples/chat_example2.png)
![蒙娜丽莎狗样例](examples/chat_example3.png)

</details>

## 友情链接

* [XrayGLM](https://github.com/WangRongsheng/XrayGLM) 是基于visualGLM-6B在X光诊断数据集上微调的X光诊断问答的项目，能根据X光片回答医学相关询问。
<details>
<summary>点击查看样例</summary>

![样例](https://github.com/WangRongsheng/XrayGLM/raw/main/assets/images/xrayglm.png)
</details>

## 使用

### 模型推理

使用pip安装依赖
```
pip install -i https://pypi.org/simple -r requirements.txt
# 国内请使用aliyun镜像，TUNA等镜像同步最近出现问题，命令如下
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```
此时默认会安装`deepspeed`库（支持`sat`库训练），此库对于模型推理并非必要，同时部分Windows环境安装此库时会遇到问题。
如果想绕过`deepspeed`安装，我们可以将命令改为
```
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements_wo_ds.txt
pip install -i https://mirrors.aliyun.com/pypi/simple/ --no-deps "SwissArmyTransformer>=0.3.6"
```

如果使用Huggingface transformers库调用模型（**也需要安装上述依赖包！**），可以通过如下代码（其中图像路径为本地路径）：
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()
image_path = "your image path"
response, history = model.chat(tokenizer, image_path, "描述这张图片。", history=[])
print(response)
response, history = model.chat(tokenizer, image_path, "这张图片可能是在什么场所拍摄的？", history=history)
print(response)
```
以上代码会由 `transformers` 自动下载模型实现和参数。完整的模型实现可以在 [Hugging Face Hub](https://huggingface.co/THUDM/visualglm-6b)。如果你从 Hugging Face Hub 上下载模型参数的速度较慢，可以从[这里](https://cloud.tsinghua.edu.cn/d/43ffb021ca5f4897b56a/)手动下载模型参数文件，并从本地加载模型。具体做法请参考[从本地加载模型](https://github.com/THUDM/ChatGLM-6B#%E4%BB%8E%E6%9C%AC%E5%9C%B0%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B)。关于基于 transformers 库模型的量化、CPU推理、Mac MPS 后端加速等内容，请参考 [ChatGLM-6B 的低成本部署](https://github.com/THUDM/ChatGLM-6B#%E4%BD%8E%E6%88%90%E6%9C%AC%E9%83%A8%E7%BD%B2)。

如果使用SwissArmyTransformer库调用模型，方法类似，可以使用环境变量`SAT_HOME`决定模型下载位置。在本仓库目录下：
```python
import argparse
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
from model import chat, VisualGLMModel
model, model_args = VisualGLMModel.from_pretrained('visualglm-6b', args=argparse.Namespace(fp16=True, skip_init=True))
from sat.model.mixins import CachedAutoregressiveMixin
model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
image_path = "your image path or URL"
response, history, cache_image = chat(image_path, model, tokenizer, "描述这张图片。", history=[])
print(response)
response, history, cache_image = chat(None, model, tokenizer, "这张图片可能是在什么场所拍摄的？", history=history, image=cache_image)
print(response)
```
使用`sat`库也可以轻松进行进行参数高效微调。<!-- TODO 具体代码 -->

## 模型微调

多模态任务分布广、种类多，预训练往往不能面面俱到。
这里我们提供了一个小样本微调的例子，使用20张标注图增强模型回答“背景”问题的能力。

解压`fewshot-data.zip`以后运行如下命令：
```
bash finetune/finetune_visualglm.sh
```

目前支持三种方式的微调：

* LoRA：样例中为ChatGLM模型的第0层和第14层加入了rank=10的LoRA微调，可以根据具体情景和数据量调整`--layer_range`和`--lora_rank`参数。
* QLoRA：如果资源有限，可以考虑使用`bash finetune/finetune_visualglm_qlora.sh`，QLoRA将ChatGLM的线性层进行了4-bit量化，只需要9.8GB显存即可微调。
* P-tuning：可以将`--use_lora`替换为`--use_ptuning`，不过不推荐使用，除非模型应用场景非常固定。

训练好以后可以使用如下命令推理：

```
python cli_demo.py --from_pretrained your_checkpoint_path --prompt_zh 这张图片的背景里有什么内容？
```

<details>
<summary>微调前后的效果对比</summary>

微调前：

```
欢迎使用 VisualGLM-6B 模型，输入图像URL或本地路径读图，继续输入内容对话，clear 重新开始，stop 终止程序
请输入图像路径或URL（回车进入纯文本对话）： fewshot-data/2p.png
用户：这张图片的背景里有什么内容？
VisualGLM-6B：人们总是被生活琐事所困扰，无法摆脱束缚.
用户：clear
请输入图像路径或URL（回车进入纯文本对话）： fewshot-data/meme.png
用户：这张图片的背景里有什么内容？
VisualGLM-6B：这张图片的背景是一张狗的图片，它躺在木地板上。
用户：clear
请输入图像路径或URL（回车进入纯文本对话）： fewshot-data/passport.png
用户：这张图片的背景里有什么内容？
VisualGLM-6B：这张图片的背景是一台电脑键盘和两张护照。
用户：clear
请输入图像路径或URL（回车进入纯文本对话）：
```

微调后：

```
欢迎使用 VisualGLM-6B 模型，输入图像URL或本地路径读图，继续输入内容对话，clear 重新开始，stop 终止程序
请输入图像路径或URL（回车进入纯文本对话）： fewshot-data/2p.png
用户：这张图片的背景里有什么内容？
VisualGLM-6B：这张图片的背景是雨天，
用户：描述一下这张图片。
VisualGLM-6B：这张图片描绘了雨天的情景，雨水淅淅沥沥地落在地面上。人们穿着雨衣或雨伞，走在雨中，似乎在寻找着什么.
用户：clear
请输入图像路径或URL（回车进入纯文本对话）： fewshot-data/meme.png
用户：这张图片的背景里有什么内容？
VisualGLM-6B：这张图片的背景是木质地板。
用户：什么颜色的地板
VisualGLM-6B：这张图片的背景是蓝色的。
用户：描述一下这张图片。
VisualGLM-6B：这张图片描绘了一只棕色的小狗躺在蓝色的木质地板上。
用户：clear
请输入图像路径或URL（回车进入纯文本对话）： fewshot-data/passport.png
用户：这张图片的背景里有什么内容？
VisualGLM-6B：这张图片的背景是电脑键盘。
用户：描述一下这张图片。
VisualGLM-6B：这张图片的特点是电脑键盘和护照。
用户：有几张护照    
VisualGLM-6B：两张护照。
用户：clear
请输入图像路径或URL（回车进入纯文本对话）：
```

</details>

如果希望把LoRA部分的参数合并到原始的权重，可以调用`merge_lora()`，例如：

```python
from finetune_visualglm import FineTuneVisualGLMModel
import argparse

model, args = FineTuneVisualGLMModel.from_pretrained('checkpoints/finetune-visualglm-6b-05-19-07-36',
        args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
        device='cuda',
    ))
model.get_mixin('lora').merge_lora()
args.layer_range = []
args.save = 'merge_lora'
args.mode = 'inference'
from sat.training.model_io import save_checkpoint
save_checkpoint(1, model, None, None, args)
```

微调需要安装`deepspeed`库，目前本流程仅支持linux系统，更多的样例说明和Windows系统的流程说明将在近期完成。

## 部署工具

### 命令行 Demo

```shell
python cli_demo.py 
```
程序会自动下载sat模型，并在命令行中进行交互式的对话，输入指示并回车即可生成回复，输入 clear 可以清空对话历史，输入 stop 终止程序。

![cli_demo](examples/thu.png)
程序提供如下超参数控制生成过程与量化精度：
```
usage: cli_demo.py [-h] [--max_length MAX_LENGTH] [--top_p TOP_P] [--top_k TOP_K] [--temperature TEMPERATURE] [--english] [--quant {8,4}]

optional arguments:
  -h, --help            show this help message and exit
  --max_length MAX_LENGTH
                        max length of the total sequence
  --top_p TOP_P         top p for nucleus sampling
  --top_k TOP_K         top k for top k sampling
  --temperature TEMPERATURE
                        temperature for sampling
  --english             only output English
  --quant {8,4}         quantization bits
```
需要注意的是，在训练时英文问答对的提示词为`Q: A:`，而中文为`问：答：`，在网页demo中采取了中文的提示，因此英文回复会差一些且夹杂中文；如果需要英文回复，请使用`cli_demo.py`中的`--english`选项。

我们也提供了继承自`ChatGLM-6B`的打字机效果命令行工具，此工具使用Huggingface模型：
```shell
python cli_demo_hf.py
```

### 网页版 Demo
![web_demo](examples/web_demo.png)

我们提供了一个基于 [Gradio](https://gradio.app) 的网页版 Demo，首先安装 Gradio：`pip install gradio`。
然后下载并进入本仓库运行`web_demo.py`：

```
git clone https://github.com/THUDM/VisualGLM-6B
cd VisualGLM-6B
python web_demo.py
```
程序会自动下载 sat 模型，并运行一个 Web Server，并输出地址。在浏览器中打开输出的地址即可使用。


我们也提供了继承自`ChatGLM-6B`的打字机效果网页版工具，此工具使用 Huggingface 模型，启动后将运行在`:8080`端口上：
```shell
python web_demo_hf.py
```

两种网页版 demo 均接受命令行参数`--share`以生成 gradio 公开链接，接受`--quant 4`和`--quant 8`以分别使用4比特量化/8比特量化减少显存占用。

### API部署
首先需要安装额外的依赖 `pip install fastapi uvicorn`，然后运行仓库中的 [api.py](api.py)：
```shell
python api.py
```
程序会自动下载 sat 模型，默认部署在本地的 8080 端口，通过 POST 方法进行调用。下面是用`curl`请求的例子，一般而言可以也可以使用代码方法进行POST。
```shell
echo "{\"image\":\"$(base64 path/to/example.jpg)\",\"text\":\"描述这张图片\",\"history\":[]}" > temp.json
curl -X POST -H "Content-Type: application/json" -d @temp.json http://127.0.0.1:8080
```
得到的返回值为
```
  {
    "response":"这张图片展现了一只可爱的卡通羊驼，它站在一个透明的背景上。这只羊驼长着一张毛茸茸的耳朵和一双大大的眼睛，它的身体是白色的，带有棕色斑点。",
    "history":[('描述这张图片', '这张图片展现了一只可爱的卡通羊驼，它站在一个透明的背景上。这只羊驼长着一张毛茸茸的耳朵和一双大大的眼睛，它的身体是白色的，带有棕色斑点。')],
    "status":200,
    "time":"2023-05-16 20:20:10"
  }
```

我们也提供了使用Huggingface模型的 [api_hf.py](api_hf.py)，用法和sat模型的api一致：
```shell
python api_hf.py
```


## 模型量化
在Huggingface实现中，模型默认以 FP16 精度加载，运行上述代码需要大概 15GB 显存。如果你的 GPU 显存有限，可以尝试以量化方式加载模型。
使用方法如下：
```python
# 按需修改，目前只支持 4/8 bit 量化。下面将只量化ChatGLM，ViT 量化时误差较大
model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).quantize(8).half().cuda()
```

在sat实现中，需先传参将加载位置改为`cpu`，再进行量化。方法如下，详见`cli_demo.py`：
```python
from sat.quantization.kernels import quantize
model = quantize(model.transformer, args.quant).cuda()
# 指定 model.transformer 只量化 ChatGLM，ViT 量化时误差较大
```

## 局限性
本项目正处于V1版本视觉和语言模型的参数、计算量都较小，我们总结了如下主要存在的改进方向：
- 图像描述事实性/模型幻觉问题。在生成图像长描述的时候，距离图像较远时，语言模型的将占主导，有一定可能根据上下文生成并不存在于图像的内容。
- 属性错配问题。在多物体的场景中，部分物体的某些属性，经常被错误安插到其他物体上。
- 分辨率问题。本项目使用了224*224的分辨率，也是视觉模型中最为常用的尺寸；然而为了进行更细粒度的理解，更大的分辨率和计算量是必要的。
- 由于数据等方面原因，模型暂时不具有中文ocr的能力（英文ocr能力有一些），我们会在后续版本中增加这个能力。
## 协议

本仓库的代码依照 [Apache-2.0](LICENSE.txt) 协议开源，VisualGLM-6B 模型的权重的使用则需要遵循 [Model License](MODEL_LICENSE.txt)。

## 引用与致谢
如果你觉得我们的工作有帮助的话，请考虑引用下列论文
```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
@article{ding2021cogview,
  title={Cogview: Mastering text-to-image generation via transformers},
  author={Ding, Ming and Yang, Zhuoyi and Hong, Wenyi and Zheng, Wendi and Zhou, Chang and Yin, Da and Lin, Junyang and Zou, Xu and Shao, Zhou and Yang, Hongxia and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={19822--19835},
  year={2021}
}
```
在VisualGLM-6B的指令微调阶段的数据集中，包含了来自[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)和[LLAVA](https://github.com/haotian-liu/LLaVA)项目的一部分英文图文数据，以及许多经典的跨模态工作数据集，衷心感谢他们的贡献。
