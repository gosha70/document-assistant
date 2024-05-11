# Document Assistant
## D.O.T. (Document Of Things)

The `Document Assistant` project is a RAG (Retrieval-Augmented Generation) end-to-end framework designed to merge private vector stores with unstructured documents and LLMs, facilitating a locally runnable Chat/QA application. It  utilizes **LangChain**, **ChromaDB**, and **PyTorch**, and is compatible with multiple document formats. 

The application's setup involves creating a vector store and transforming documents into textual splits. The project's core is its ability to create a private, customized database for enhanced retrieval and generation tasks, demonstrating a significant advancement in leveraging LLMs for personalized and efficient data handling​.

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


### <a name="quickStart">Quick Sstart Guide</a>

#### Create Vector Store 
1. The first step is the creation vectorstore for your documents. 
Just scan the document folder and create a vectorstore:
```
python3 -m embeddings.embedding_database --dir_path ~/ --file_types pdf --persist_directory pdf_db
```
Where:
- `--dir_path` - the path to the document folder where desired documents are located  
- `--file_types`-  one or a few supported file types:   
    - **CSV** = `"csv"`
    - **DDL** = `"ddl"`
    - **EXCEL** = `"xlsx"`
    - **JAVA** = `"java"`
    - **JS** = `"js"`
    - **JSON**  = `"json"`
    - **HTML** = `"html"`
    - **MARKDOWN** = `"md"`
    - **PDF** = `"pdf"`
    - **PYTHON** = `"py"`
    - **RICH_TEXT** = `"rtf"`
    - **SQL** = `"sql"`
    - **TEXT** = `"txt"`
    - **XML** = `"xml"`
    - **XSL** = `"xsl"`
    - **YAML** = `"yaml"`
           
- `--persist_director` - the location where a vectorstore will be created; in this example, the vectorstore will be create in the `pdf_db` folder.

2. Run the **D.O.T.** Application with **RAG** based on your vectorstore
```
python3 -m app.chat_app --persist_directory pdf_db --history True
```

### Detailed Instructions
#### Creating Vector Store
See [Chroma](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html)

##### Seach and Transform Documents to Textual Splits
See [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)
```
python3 -m embeddings.document_loader --dir_path ~/ --file_types java
```

##### Creating Vector Store with Embedding
```
python3 -m embeddings.embedding_database --dir_path ~/ --file_types pdf --persist_directory pdf_db
```
The vector store can also be created via the [Colab Notebook](https://github.com/gosha70/document-assistant/blob/main/notebooks/CreateEmbeddingsVectorstore.ipynb)
:
1. Unstructured document splits can be created in a local machine:
```
python3 -m embeddings.document_loader --dir_path ~/ --file_types java --persist_directory java_splits
```


2. Zip a folder with document splits, and upload that zip to your Google Drive
3. Open the [Colab Notebook](https://github.com/gosha70/document-assistant/blob/main/notebooks/CreateEmbeddingsVectorstore.ipynb) in **Google Colaboratory**
4. Follow the documentation in the notebook

Similarary, the vector store can be locally created from document splits:

- From a zip file:
```
python3 -m embeddings.embedding_database --zip_file SPLITS.zip --persist_directory java_db --collection_name JAVA_DB
```
- From a directory with unstructured documents:
```
python3 -m embeddings.embedding_database --splits_directory ./splits --persist_directory java_db --collection_name JAVA_DB
```

##### Manifest
The `MANIFEST.MF` with the important information about generated vector store can be found in the `META-INF` folder in the specified `persist_director`:
```
Manifest-Version: 1.0
Created-On: 2023-12-31T14:13:33.392Z
Created-By: EGOGE (https://github.com/gosha70/document-assistant)
Collection-Name: EGOGE_DOCUMENTS_DB
Embedding-Class: langchain_community.embeddings.HuggingFaceInstructEmbeddings
Embedding-Model-Name: hkunlp/instructor-large
```
:triangular_flag_on_post: You must remember the `collection name` in order to load the created vector store; the **D.O.T. Application** defaults to `EGOGE_DOCUMENTS_DB`, otherwise the `collection name` must be passed via `--collection_name`.

### Demo 

Here is the example of creating the private chat (Q/A) about the Java framework for simulation the `Quantum Computing`; the source for this can be found [here](https://www.manning.com/books/quantum-computing-in-action). 

After the source was downloaded, run the following command to create the vector store from processing **Java** and **Markdown** files:
```
python3 -m embeddings.embedding_database --dir_path c --file_types java md --persist_directory ~./quantumjava_doc_db
```

:movie_camera: [Creating Chroma for Java Examples of Quantum Computing](https://drive.google.com/file/d/19xbVvnTvkTTV4lZ3rcFIe08WJn3hMLbG)


###  D.O.T. Appicatiion
The `Document Assistant` project provides the out-of-box **Dash** application which hosts a generic LLM with a private vector store.

![image](https://github.com/gosha70/document-assistant/assets/17832712/76847401-8c27-4e2b-a613-218c622ba395)

The explanation about the parameters for starting the application and customizing runtime **RAG** can be found at the `main` method of [chat_app](https://github.com/gosha70/document-assistant/blob/main/app/chat_app.py).

:triangular_flag_on_post: You must remember the `collection name` in order to load the created vector store; the **D.O.T. Application** defaults to `EGOGE_DOCUMENTS_DB`, otherwise the `collection name` must be passed via `--collection_name`.

:information_source: Pass `--history True` if the chat uses the previously asked questions and answers.

#### Customizing Application

##### Content Customization
UX of D.O.T. Appicatiion can be easily customized via [App Config JSON](https://github.com/gosha70/document-assistant/blob/main/app/app_config.json).
For example, see how it is done in the demo application for [Quantum Computing Q/A](https://github.com/gosha70/quantum-chat/blob/main/app/app_config.json)

##### Color Scheme
The default dark mode color sheme can be switched to the out-of-box color schemes (for example, see how it is done in [Quantum Computing Q/A](https://github.com/gosha70/quantum-chat)).
In order to use a different color scheme, replace `*_color-scheme.css` in the [assets folder](https://github.com/gosha70/document-assistant/tree/main/app/assets) with your `CSS` with overwritten properties, or with one in the [color_schemes folder](https://github.com/gosha70/document-assistant/tree/main/app/color_schemes).



#### Demo Appicatiion
1. `pip3 install dash dash-bootstrap-components`
2. `python3 -m app.chat_app --persist_directory ./quantumjava_doc_db --history True`

:movie_camera: [Quantum Computing Q/A](https://drive.google.com/file/d/1OxAUQoNFPsGm9yvhaencaYZCxuqhlcL6) 


