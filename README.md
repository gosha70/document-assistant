#### Requirements

##### [LangChain](https://python.langchain.com/docs/modules/chains/foundational/llm_chain)

##### [ChromaDB](https://docs.trychroma.com/)

##### [PyTorch](https://pytorch.org/get-started/locally)

##### GPTQ
###### [AutoGPTQ](https://pypi.org/project/auto-gptq/) 
If the installation or execution of **AutoGPTQ** fails, please follow this [workround](https://huggingface.co/TheBloke/WizardLM-30B-Uncensored-GPTQ).

#### Genrating Embedding Database
```
python3 -m embeddings.embedding_database --dir_path ~/ --file_types pdf --persist_directory pdf_db
```

#### Demo Appicatiion
1. 'pip3 install dash dash-bootstrap-components'
2. `python3 -m app.chat_app --persist_directory pdf_dbpdf_db`

