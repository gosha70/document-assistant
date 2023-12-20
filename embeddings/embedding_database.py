import argparse
import logging
from langchain.vectorstores import Chroma

from .document_loader import load_documents

from embeddings.embeddings_constants import CHROMA_SETTINGS

from models.model_info import ModelInfo

from .document_loader import get_FileLoaderQuery

"""
Creates a (Chroma) embedding database which stores processed unstructured document splits.

Parameters:
- documents (List[str]): The unstructured document splits
- model_name ({str}): The embedding model name
- persist_directory ({str}): The optional file path to store the  embedding database; 
                             if it is not specified, (Chroma) is not persisted.

Returns:
- (Chroma): the embedding database
"""
def create_embedding_database(documents, model_name, persist_directory) -> Chroma:
    if documents is None or len(documents) == 0:
        logging.warning(f"Cannot create an embedding database from empty list of documents!")  
        return None  

    embedding = ModelInfo.create_embedding(model_name=model_name)

    logging.info(f"Creating the embedding database with {embedding} for {len(documents)} document splits ...")    
    docs_db = Chroma.from_documents(documents, embedding=embedding, persist_directory=persist_directory)
    if persist_directory is not None:
        logging.info(f"Saving the embedding database in {persist_directory} ...")    
        docs_db.persist()

    return docs_db

"""
Creates a (Chroma) embedding database which stores processed unstructured document splits
associates with the specified file types.

Parameters:
- dir_path (str): The root directory where the search for documents is performed
- file_loader_query (FileType): The file type to load
- persist_directory ({str}): The optional file path to store the  embedding database; 
                             if it is not specified, (Chroma) is not persisted.
- model_name ({str}): The embedding model name; if it is not specified, it is set to "BAAI/bge-small-en"

Returns:
- (Chroma): the embedding database if documents are found and processed; otherwise - None.
"""
def create_vector_store(dir_path, file_loader_query, persist_directory, model_name) -> Chroma:
    # Load documents
    split_docs = load_documents(dir_path, file_loader_query)

    if split_docs is not None:
        # Create embeddings and database
        return create_embedding_database(split_docs, model_name, persist_directory)
    
    return None


"""
Load the Chroma for the vectorstore persisted in the specified directory.

Parameters:
- model_name (str): The embedding model name
- persist_directory ({str}): The optional file path to store the  embedding database; 
                             if it is not specified, (Chroma) is not persisted.

Returns:
- (Chroma): the embedding database if documents are found and processed; otherwise - None.
"""
def load_vector_store(model_name, persist_directory) -> Chroma:
    logging.info(f"Creating (HuggingFaceInstructEmbeddings) for '{model_name}' ...")
    
    embedding = ModelInfo.create_embedding(model_name=model_name)

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        client_settings=CHROMA_SETTINGS,
    )


if __name__ == "__main__":
    # Set the logging level to INFO
    logging.basicConfig(level=logging.INFO)

    # Create the parser
    parser = argparse.ArgumentParser(description="Creating the embedding database.")

    # Add the arguments
    parser.add_argument('--dir_path', type=str, help='Root directory where to look for documents.', default=".")
    parser.add_argument('--file_types', type=str, nargs='+', help='List of file extensions without the do: md; java; xml; html; pdf')
    parser.add_argument('--file_patterns', nargs='+', help='Name patterns for each file type; for example: --file_patterns "java:**/*Function* html:**/*"', default=[])   
    parser.add_argument('--persist_directory', type=str, help='(Optional) The path to the directory where the embedding database will be persisted.', default=None)
    parser.add_argument('--model_name', type=str, help='The name of embedding model for analyzing unstructured document splits.', default=None)
    parser.add_argument('--test_question', type=str, help='(Optional) Indicates if the created vectordatastore should be tested.', default=None)
      
    # Parse the arguments
    args = parser.parse_args()

    logging.info(f"Creating the embedding database with the arguments: {args}")

    file_loader_query = get_FileLoaderQuery(args.file_types, args.file_patterns)

    # Call the create_vector_store function
    docs_db = create_vector_store(args.dir_path, file_loader_query, args.persist_directory, args.model_name)

    if args.test_question is not None:
        retriever = docs_db.as_retriever()
        answer = retriever.invoke(args.test_question)
        logging.info(f"ANSWER: {answer[0].page_content}")

    # Optionally, you can print or log the result
    logging.info(f"The embedding database: {docs_db}")