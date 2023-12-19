from transformers import AutoModelForCausalLM, AutoTokenizer

from .models_constants import DEVICE_MAP

"""
Creates a AWQ (Activation-aware Weight Quantization) quantization method which is a simple 
method for quantizing (compressing) Large Language Models ( LLMs ) to reduce their runtime 
and storage requirements for inference.

This function relies on AutoGPTQForCausalLM.

Parameters:
- model_info (ModelInfo): the class storing the information about LLM:
     model_name (str) 
     model_id (str) 
     model_basename (str) 
     device_type (str)
Returns:
- tokenizer (AutoTokenizer): The tokenizer supporting the returned quantized model.
- model (AutoModelForCausalLM): The quantized model.
"""
def load_gptq_model(model_info):    
    tokenizer = AutoTokenizer.from_pretrained(
        model_info.model_id, 
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_info.model_id,
        use_safetensors=True,
        trust_remote_code=True,
        device_map=DEVICE_MAP,
    )
    return tokenizer, model