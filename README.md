### Installation
1. Install required Python libraries
```
pip3 install -r requirements.txt
```
#### Requirement Details

- [LangChain](https://python.langchain.com/docs/modules/chains/foundational/llm_chain)

- [ChromaDB](https://docs.trychroma.com/)

- [PyTorch](https://pytorch.org/get-started/locally)

- [AutoGPTQ](https://pypi.org/project/auto-gptq/) 
By default the **AutoGPTQ** (`auto-gptq`) is not installed. Before installing it, please configure the [CUDA](https://www.cs.colostate.edu/~info/cuda-faq.html) environment first. 
If the installation or execution of **AutoGPTQ** fails, please follow this [workaround](https://huggingface.co/TheBloke/WizardLM-30B-Uncensored-GPTQ).

### Creating Vector Store
See [Chroma](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html)

#### Seach and Transform Documents to Textual Splits
See [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)
```
python3 -m embeddings.document_loader --dir_path ~/ --file_types java
```

#### Creating Vector Store with Embedding
```
python3 -m embeddings.embedding_database --dir_path ~/ --file_types pdf --persist_directory pdf_db
```
The vectorestor can also be created via the [Colab Notebook](https://github.com/gosha70/document-assistant/blob/main/notebooks/CreateEmbeddingsVectorstore.ipynb)
:
1. Unstructured document splits can be created in a local machine:
```
python3 -m embeddings.document_loader --dir_path ~/ --file_types java --persist_directory java_splits
```
2. Zip a folder with document splits, and upload that zip to your Google Drive
3. Open the [Colab Notebook](https://github.com/gosha70/document-assistant/blob/main/notebooks/CreateEmbeddingsVectorstore.ipynb) in **Google Colaboratory**
4. Follow the documentation in the notebook

Similarary, the vectorestor can be locally created from a zip with document splits:
```
python3 -m embeddings.embedding_database --zip_file SPLITS.zip --persist_directory java_db --collection_name JAVA_DB
```

### Demo Appicatiion
1. `pip3 install dash dash-bootstrap-components`
2. `python3 -m app.chat_app --persist_directory pdf_dbpdf_db`

