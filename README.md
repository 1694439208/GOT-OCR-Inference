# GOT-OCR-Inference
研究GOT-OCR-项目落地加速，不限语言

> ## 研究1:
- [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [llama-cpp-python](https://github.com/Ucas-HaoranWei/GOT-OCR2.0](https://github.com/abetlen/llama-cpp-python))
- [release-exe](https://huggingface.co/kaifeise/GOT-gguf/tree/main)

> ## release
- 基础sdk包： https://pan.baidu.com/s/10Lo-yY_ZNW7gs0Gd9hiaMw 提取码: ie4n
- 更新包: https://pan.baidu.com/s/1pw2JRQZjBZYo4UU-7UNuhQ 提取码: 5x3d
- 下载基础sdk包解压，然后下载更新包覆盖解压，然后  **双击启动.bat** 启动

- 由于很多不熟悉got，想快速应用，现有偿提供release包的源码

| 费用 | 项目源码 | 项目解惑 |
|-----|-----|-----|
| 500r | ✓ | × |
| 999r | ✓ | ✓ |

```
代码里直接使用
<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
<img></img>
OCR: <|im_end|><|im_start|>assistant提示词是为了测试代码是否正常，主要是为了测试嵌入向量

pip install llama-cpp-python
研究GOT-OCR2.0落地加速，经过查询llama-cpp-python和llama的源码demo和issues，暂时实现了可能的推理，因为他官方就没说过也没找到如何嵌入自定义向量

量化后的模型，不保证对，因为是直接基于官方提供的模型做的量化，可能会有got的层被量化进来
通过百度网盘分享的文件：None-619M-123-F16.gguf
链接：https://pan.baidu.com/s/1nWkMVrwPcb1qjkGTSz4g6g 
提取码：3zop

如果要自己量化可以参考使用我修改的*convert_hf_to_gguf.py*脚本
config.json文件的
"architectures": [
  "GOTQwenForCausalLM"
],
要改成
"architectures": [
  "Qwen2ForCausalLM"
],
否则量化脚本找不到模型结构类型
```
