import ctypes
import os
import multiprocessing
import struct

import llama_cpp
import torch

import numpy as np
"""
模型保存成功
<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
<img><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad></img>
OCR: <|im_end|><|im_start|>assistant
"""
#pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

def float_list_to_ctypes_array(float_list):
    # 创建一个ctypes的float数组类型
    FloatArray = ctypes.c_float * len(float_list)
    
    # 使用这个类型创建一个新的ctypes数组，并用float_list初始化它
    return FloatArray(*float_list)


MODEL_PATH = os.environ.get("MODEL", r"D:\csx_demo\GOT-OCR2.0\GOT-OCR-2.0-master\GOT_weights\None-619M-123-F16.gguf")


llm = llama_cpp.Llama(
      model_path=MODEL_PATH,
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=3768, # 根据电脑内存自行设置
)
llm.eval(llm.tokenize(B"""<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user"""))
#cparams.n_ctx = 5768

# 读取保存的张量#generate
tensor = torch.load('tensor.pt')
# 将张量转换为 NumPy 数组
embd_image = torch.tensor(tensor).squeeze().cpu().tolist()#.to(torch.float32)
# 将 NumPy 数组转换为 list[ctypes.c_int32]
#embd_inp = [ctypes.c_int32(int(x)) for x in embd_inp]
#embd_inp = [ctypes.c_int32(struct.unpack('i', struct.pack('f', x))[0]) for x in embd_inp]
import llama_cpp.llava_cpp as llava_cpp

n_past = ctypes.c_int(llm.n_tokens)
n_past_p = ctypes.pointer(n_past)

#官方传的是图片然后用他的预处理模型转向量，这里我们需要自己构建一个向量结构，
c_float_array = float_list_to_ctypes_array(embd_image)
embed = llava_cpp.llava_image_embed(embed=c_float_array,n_image_pos=2)
#n_image_pos=2 是 len(c_float_array) / llm.n_batch  此模型的n_batch是512  而向量是1024

with llama_cpp.suppress_stdout_stderr(disable=False):
    llava_cpp.llava_eval_image_embed(
        llm.ctx,
        embed,
        llm.n_batch,
        n_past_p,
    )
# Required to avoid issues with hf tokenizer
llm.input_ids[llm.n_tokens : n_past.value] = -1
llm.n_tokens = n_past.value


"""output = llm(
      "OCR: <|im_end|><|im_start|>assistant", # Prompt
      #max_tokens=100, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
for item in output["choices"]:
    print(item["text"])"""



tokens = llm.tokenize(b"OCR: <|im_end|><|im_start|>assistant")
for token in llm.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.0):
    print(llm.detokenize([token]),end="")


llm.close()