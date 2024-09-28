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

def eval_tokens(ctx_llama:llama_cpp.llama_context_p, tokens:list[llama_cpp.llama_token], n_batch:int, n_past:llama_cpp.llama_pos):
    N = len(tokens)
    for i in range(0, N, n_batch):
        n_eval = N - i
        if n_eval > n_batch:
            n_eval = n_batch

        # Convert Python list to ctypes array
        c_tokens = (ctypes.c_int * n_eval)(*tokens[i:i+n_eval])
        
        # Assuming llama_batch_get_one returns a batch of tokens
        batch = llama_cpp.llama_batch_get_one(c_tokens, n_eval, n_past, 0)
        
        if llama_cpp.llama_decode(ctx_llama, batch):
            print(f"{__name__}: failed to eval. token {i}/{N} (batch size {n_batch}, n_past {n_past})")
            return False
        
        n_past.value += n_eval
    
    return True

MODEL_PATH = os.environ.get("MODEL", r"D:\csx_demo\GOT-OCR2.0\GOT-OCR-2.0-master\GOT_weights\None-464M-123-F16.gguf")


llm = llama_cpp.Llama(
      model_path=MODEL_PATH,
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=3768, # 根据电脑内存自行设置
      n_batch=1024,
      n_ubatch=1024,
)


#cparams.n_ctx = 5768

# 读取保存的张量#generate
tensor = torch.load('tensora.pt')
# 将张量转换为 NumPy 数组
embd_image = tensor.squeeze().cpu().tolist()#.to(torch.float32)
# 将 NumPy 数组转换为 list[ctypes.c_int32]
#embd_inp = [ctypes.c_int32(int(x)) for x in embd_inp]
#embd_inp = [ctypes.c_int32(struct.unpack('i', struct.pack('f', x))[0]) for x in embd_inp]
import llama_cpp.llava_cpp as llava_cpp

n_past = ctypes.c_int(llm.n_tokens)
n_past_p = ctypes.pointer(n_past)

#官方传的是图片然后用他的预处理模型转向量，这里我们需要自己构建一个向量结构，
c_float_array = float_list_to_ctypes_array(embd_image)
embed = llava_cpp.llava_image_embed(embed=c_float_array,n_image_pos=1)
#n_image_pos=2 是 len(c_float_array) / llm.n_batch  此模型的n_batch是512  而向量是1024


#star
ret = eval_tokens(ctx_llama = llm.ctx,tokens=llm.tokenize(B"""<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user""",True,True),
n_batch=llm.n_batch,
n_past= n_past)


with llama_cpp.suppress_stdout_stderr(disable=True):
    llava_cpp.llava_eval_image_embed(
        llm.ctx,
        embed,
        llm.n_batch,
        n_past_p,
    )

ret = eval_tokens(ctx_llama = llm.ctx,tokens=llm.tokenize(B"OCR: <|im_end|><|im_start|>assistant",False,True),n_batch=llm.n_batch,n_past= n_past)
# Required to avoid issues with hf tokenizer
#llm.input_ids[llm.n_tokens : n_past.value] = -1
#llm.n_tokens = 2#n_past.value
#print(f"llm.n_tokens:{llm.n_tokens}")

# Required to avoid issues with hf tokenizer
llm.input_ids[llm.n_tokens : n_past.value] = -1
llm.n_tokens = n_past.value

# Get prompt tokens to avoid a cache miss
prompt = llm.input_ids[: llm.n_tokens].tolist()

"""output = llm(
      "OCR", # Prompt
      max_tokens=3500, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
for item in output["choices"]:
    print(item["text"])"""


"""for _ in range(200):
    print(llm.detokenize([llm.sample()]).decode(),end="")"""

tokens = llm.tokenize(b"O")
for token in llm.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.0):
    print(llm.detokenize([token]),end="")


llm.close()