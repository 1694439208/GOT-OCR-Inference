import ctypes
import os
import multiprocessing
import struct

import llama_cpp
import torch

import numpy as np

import llama_cpp.llava_cpp as llava_cpp
"""
模型保存成功
<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
<img><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad><imgpad></img>
OCR: <|im_end|><|im_start|>assistant
"""
#pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
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

def float_list_to_ctypes_array(float_list):
    # 创建一个ctypes的float数组类型
    FloatArray = ctypes.c_float * len(float_list)
    
    # 使用这个类型创建一个新的ctypes数组，并用float_list初始化它
    return FloatArray(*float_list)

llama_cpp.llama_backend_init(numa=False)

N_THREADS = multiprocessing.cpu_count()
MODEL_PATH = os.environ.get("MODEL",b"D:\csx_demo\GOT-OCR2.0\GOT-OCR-2.0-master\GOT_weights\None-464M-123-F16.gguf")

promptaaa = b"""
<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
<img></img>
OCR: <|im_end|><|im_start|>assistant
"""

lparams = llama_cpp.llama_model_default_params()
cparams = llama_cpp.llama_context_default_params()
cparams.n_ctx = 5768

model = llama_cpp.llama_load_model_from_file(MODEL_PATH, lparams)
ctx = llama_cpp.llama_new_context_with_model(model, cparams)

n_past = ctypes.c_int(0)
n_past_p = ctypes.pointer(n_past)

data = B"""<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
"""
embd_inp = (llama_cpp.llama_token * (len(data) + 1))()
n_of_tok = llama_cpp.llama_tokenize(
    model,
    data,
    len(embd_inp),
    embd_inp,
    len(embd_inp),
    False,
    False,
)
embd_inp = embd_inp[:n_of_tok]

ret = eval_tokens(ctx_llama = ctx,tokens=embd_inp,n_batch=512,n_past= n_past)



# 读取保存的张量#generate
tensor = torch.load('tensora.pt')

aa = torch.tensor(tensor).shape
# 将张量转换为 NumPy 数组
embd_image = torch.tensor(tensor).squeeze().cpu().tolist()#.to(torch.float32)
# 将 NumPy 数组转换为 list[ctypes.c_int32]
#embd_inp = [ctypes.c_int32(int(x)) for x in embd_inp]
#embd_inp = [ctypes.c_int32(struct.unpack('i', struct.pack('f', x))[0]) for x in embd_inp]
c_float_array = float_list_to_ctypes_array(embd_image)
embed = llava_cpp.llava_image_embed(embed=c_float_array,n_image_pos=2)
with llama_cpp.suppress_stdout_stderr(disable=True):
    llava_cpp.llava_eval_image_embed(
        ctx,
        embed,
        512,
        n_past_p,
    )
"""batch = llama_cpp.llama_batch(
            0,
            None,
            c_float_array,
            None,
            None,
            None,
            None,
            n_past,
            1,
            0
        )
if llama_cpp.llama_decode(ctx, batch) != 0:
    print("Error decoding")"""



n_ctx = llama_cpp.llama_n_ctx(ctx)

n_predict = 1024

input_consumed = 0
input_noecho = False

remaining_tokens = n_predict

embd = []
last_n_size = 64
last_n_tokens_data = [0] * last_n_size
n_batch = 512
last_n_repeat = 64
repeat_penalty = 1
frequency_penalty = 0.0
presence_penalty = 0.0

max_tokens = n_ctx
# use ctypes arrays for array params
data = B"""OCR: <|im_end|><|im_start|>assistant"""
embd_inp1 = (llama_cpp.llama_token * (len(data) + 1))()
n_of_tok1 = llama_cpp.llama_tokenize(
    model,
    data,
    len(embd_inp1),
    embd_inp1,
    len(embd_inp1),
    False,
    False,
)
embd_inp1 = embd_inp1[:n_of_tok1]

ret = eval_tokens(ctx_llama = ctx,tokens=embd_inp1,n_batch=512,n_past= n_past)

embd_inp += embd_inp1
while remaining_tokens > 0:

    n_past.value += len(embd)
    embd = []
    if len(embd_inp) <= input_consumed:
        logits = llama_cpp.llama_get_logits(ctx)
        n_vocab = llama_cpp.llama_n_vocab(model)

        _arr = (llama_cpp.llama_token_data * n_vocab)(
            *[
                llama_cpp.llama_token_data(token_id, logits[token_id], 0.0)
                for token_id in range(n_vocab)
            ]
        )
        candidates_p = llama_cpp.ctypes.pointer(
            llama_cpp.llama_token_data_array(_arr, len(_arr), False)
        )

        _arr = (llama_cpp.llama_token * len(last_n_tokens_data))(*last_n_tokens_data)
        llama_cpp.llama_sample_repetition_penalties(
            ctx,
            candidates_p,
            _arr,
            last_n_repeat,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
        )

        llama_cpp.llama_sample_top_k(ctx, candidates_p, 40,1)
        llama_cpp.llama_sample_top_p(ctx, candidates_p, 0.8, 1)
        #llama_cpp.llama_sample_temperature(ctx, candidates_p,0.2)
        id = llama_cpp.llama_sample_token(ctx, candidates_p)

        last_n_tokens_data = last_n_tokens_data[1:] + [id]
        embd.append(id)
        input_noecho = False
        remaining_tokens -= 1
    else:
        while len(embd_inp) > input_consumed:
            embd.append(embd_inp[input_consumed])
            last_n_tokens_data = last_n_tokens_data[1:] + [embd_inp[input_consumed]]
            input_consumed += 1
            if len(embd) >= n_batch:
                break
    if not input_noecho:
        for id in embd:
            size = 32
            buffer = (ctypes.c_char * size)()
            n = llama_cpp.llama_token_to_piece(
                model, llama_cpp.llama_token(id), buffer, size,0,False
            )
            assert n <= size
            print(
                buffer[:n].decode("utf-8"),
                end="",
                flush=True,
            )

    if len(embd) > 0 and embd[-1] == llama_cpp.llama_token_eos(ctx):
        break

print()

llama_cpp.llama_print_timings(ctx)

llama_cpp.llama_free(ctx)

#llama_cpp.llama_decode()

#llava_eval_image_embed