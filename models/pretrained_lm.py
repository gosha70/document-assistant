import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from langchain_community.llms import LlamaCpp

from .models_constants import (
    DEVICE_MAP, 
    DEVICE_TYPE_MPS,
    DEVICE_TYPE_CUDA
)

QUANT_TYPE="nf4"
if torch.cuda.is_available():
    MAX_MEMORY = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
else:
   MAX_MEMORY = "4GB"    

"""
Returns a pretrained model based on the specified device type.
If the device type is either 'cpu' or 'mps', LlamaForCausalLM model and 
LlamaTokenizer tokenizer supporting that model are created; otherwise, -
AutoModelForCausalLM model and AutoTokenizer tokenizer supporting that model are created.

See: https://huggingface.co/docs/transformers/model_doc/auto

Parameters:
- model_info (ModelInfo): the class storing the information about LLM:
     model_name (str) 
     model_id (str) 
     model_basename (str) 
     device_type (str)
- cache_dir (str): The path to the local cache directory where loaded models are stored.

Returns:
- LlamaCpp: The LlamaCpp model if successful, otherwise - None.
"""
def load_pretrained_model(model_info, cache_dir):    

    if model_info.device_type.lower() in [DEVICE_TYPE_MPS, DEVICE_TYPE_CUDA]:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_info.model_id, 
            cache_dir=cache_dir
        )
        model = LlamaForCausalLM.from_pretrained(
            model_info.model_id, 
            cache_dir=cache_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_info.model_id, 
            cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_info.model_id,
            device_map=DEVICE_MAP,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
            load_in_4bit=True,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.float16,
            max_memory=MAX_MEMORY
        )
        model.tie_weights()
    return tokenizer, model