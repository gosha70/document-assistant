# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import logging
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

from transformers import (
    GenerationConfig,
    pipeline,
)

# Local API
from models.models_constants import N_CTX
from models.awq_lm import load_gptq_model as awq
from models.gguf_lm import load_gguf_model as gguf
from models.gptq_lm import load_gptq_model as qptq
from models.pretrained_lm import load_pretrained_model as pretrained

from retrieval_constants import (
    AWQ_EXTENSION, 
    CACHE_DIR, 
    CHAIN_TYPE_STUFF,
    GGML_EXTENSION, 
    GGUF_EXTENSION
)

CALLBACK_MANAGER = CallbackManager([StreamingStdOutCallbackHandler()])

"""
Creates a quantized model corresponding the specified model id and file name.

Parameters:
- model_info (ModelInfo): the class storing the information about LLM:
     model_name (str) 
     model_id (str) 
     model_basename (str) 
     device_type (str)

Returns:
   - LLM: (LlamaCpp) for GGUF/GGML; otherwise - (HuggingFacePipeline)
    
See https://huggingface.co/docs/transformers/    
"""
def create_model(model_info):
    logging.info(f"Creating Model Pipeline - '{model_info.model_id}/{model_info.model_basename}' on '{model_info.device_type}'")
       
    if model_info.model_basename is not None:
        lowercaseFileName = model_info.model_basename.lower() 
        if lowercaseFileName.endswith(GGUF_EXTENSION) or lowercaseFileName.endswith(GGML_EXTENSION):
            return gguf(model_info, cache_dir=CACHE_DIR)
        elif lowercaseFileName.endswith(AWQ_EXTENSION):
            tokenizer, model = awq(model_info)
        else:
            tokenizer, model = qptq(model_info)
    else:
        tokenizer, model = pretrained(model_info, cache_dir=CACHE_DIR)

    generation_config = GenerationConfig.from_pretrained(model_info.model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=N_CTX,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.25,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Model Pipeline has been created")

    return local_llm

"""
Create the retrieval framework for the QA chat application.

The  framework uses RetrievalQA referencing HuggingFaceInstructEmbeddings and the spcified Chroma vectorstore. 

Parameters:
- model_info (map): the map storing the information about LLM:
    model_id (str) 
    model_name (str) 
    model_basename (str) 
    device_type (str)
- prompt_info (str): the promp template type: 'llama', 'mistral'
    system_prompt (str): the system prompt instructions 
    template_type (str): the promp template type: 'llama', 'mistral'
    use_history (bool): the flag indicating if the chat history is on     
- vectorstore (Chroma): the vectorstore

Returns:
- RetrievalQA: the retrieval framewor
"""
def create_retrieval_qa(model_info, prompt_info, vectorstore):

    if not isinstance(vectorstore, Chroma):
        raise TypeError("vectorstore must be of type Chroma")
        
    retriever = vectorstore.as_retriever()

    # load the LLM
    llm = create_model(model_info=model_info)

    # get the prompt template and memory if set by the user.
    prompt, memory = prompt_info.get_prompt_template()

    if prompt_info.use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=CHAIN_TYPE_STUFF,
            retriever=retriever,
            return_source_documents=True, 
            callbacks=CALLBACK_MANAGER,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=CHAIN_TYPE_STUFF,
            retriever=retriever,
            return_source_documents=True, 
            callbacks=CALLBACK_MANAGER,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa
