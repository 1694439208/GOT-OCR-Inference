import ctypes
from llama_cpp import Llama
import llama_cpp
import llama_cpp.llava_cpp as llava_cpp
import torch

def float_list_to_ctypes_array(float_list):
    # 创建一个ctypes的float数组类型
    FloatArray = ctypes.c_float * len(float_list)
    
    # 使用这个类型创建一个新的ctypes数组，并用float_list初始化它
    return FloatArray(*float_list)



llm = Llama(
      model_path=r"D:\csx_demo\GOT-OCR2.0\GOT-OCR-2.0-master\GOT_weights\None-464M-123-F16.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

n_past = ctypes.c_int(llm.n_tokens)
n_past_p = ctypes.pointer(n_past)

tensor = torch.load('tensor.pt')
# 将张量转换为 NumPy 数组
embd_image = tensor.squeeze().cpu().tolist()#.to(torch.float32)
c_float_array = float_list_to_ctypes_array(embd_image)
embed = llava_cpp.llava_image_embed(embed=c_float_array,n_image_pos=1)

with llama_cpp.suppress_stdout_stderr(disable=True):
    llava_cpp.llava_eval_image_embed(
        llm.ctx,
        embed,
        llm.n_batch,
        n_past_p,
    )

output = llm(
      """<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
OCR: <|im_end|><|im_start|>assistant""", # Prompt
      max_tokens=3032, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output["choices"][0]["text"])