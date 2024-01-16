# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models_constants import DEVICE_MAP

SAFETENSORS_EXT = ".safetensors"

def adjust_model_basename(model_basename):
    if SAFETENSORS_EXT in model_basename:
        return model_basename.replace(SAFETENSORS_EXT, "")
    return model_basename

"""
Creates a GPTQ quantization method which is a neural network compression technique that enables 
the efficient deployment of Generative Pretrained Transformers ( GPT ).

This function relies on AutoGPTQForCausalLM.

Parameters:
- model_info (ModelInfo): the class storing the information about LLM:
     model_name (str) 
     model_id (str) 
     model_basename (str) 
     device_type (str)

Returns:
- tokenizer (AutoTokenizer): The tokenizer supporting the returned quantized model.
- model (AutoGPTQForCausalLM): The quantized model.
"""
def load_gptq_model(model_info):    

    tokenizer = AutoTokenizer.from_pretrained(model_info.model_id, use_fast=True)

    model = AutoModelForCausalLM.from_quantized(
        model_info.model_id,
        model_basename=adjust_model_basename(model_info.model_basename),
        use_safetensors=True,
        trust_remote_code=True,
        device_map=DEVICE_MAP,
        use_triton=False,
        quantize_config=None,
    )
    return tokenizer, model 