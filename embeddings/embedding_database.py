import argparse
import logging
import asyncio
import time
import os
from datetime import datetime
from langchain.vectorstores import Chroma

from .document_loader import load_documents

from embeddings.embeddings_constants import CHROMA_SETTINGS, DEFAULT_COLLECTION_NAME, BATCH_SIZE, get_elapse_time_message

from models.model_info import ModelInfo
from models.models_constants import DEFAULT_MODEL_NAME

from .document_loader import get_FileLoaderQuery, load_zip_with_splits, load_document_split

def create_manifest(collection_name, model_name, persist_directory):
    if persist_directory is None:
        return False
    
    if collection_name is None:
        collection_name = DEFAULT_COLLECTION_NAME

    if model_name is None:
        model_name = DEFAULT_MODEL_NAME    
   
    meta_inf_path = os.path.join(persist_directory, 'META-INF')
    manifest_file_path = os.path.join(meta_inf_path, 'MANIFEST.MF')
    
    # Get current UTC time and format it
    current_utc_datetime = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' 

    # Manifest content
    manifest_content = f"""Manifest-Version: 1.0\nCreated-On: {current_utc_datetime}\nCreated-By: EGOGE (https://github.com/gosha70/document-assistant)\nCollection Name: {collection_name}\nEmbedding Class: {ModelInfo.embedding_class()}\nEmbedding Model Name: {model_name}
    """

    if not os.path.exists(meta_inf_path):
        os.makedirs(meta_inf_path)
        print(f"Created directory: {meta_inf_path}")

    with open(manifest_file_path, 'w') as file:
        file.write(manifest_content)
        print(f"Created MANIFEST.MF at: {manifest_file_path}")

    return True    


def on_task_completion(future, task_id, completed_tasks):
    # Update the list of completed tasks
    completed_tasks.append(task_id)
    print(f"The async task #{task_id} just finished. Total finished tasks: {len(completed_tasks)}")


async def wait_for_tasks(docs_db, async_tasks, completed_tasks) -> Chroma:
    """
    Waits for the specified async tasks to finish, then saves the vectorstoore.

    Parameters:
    - docs_db (Chroma): the vectorstore
    - async_tasks (List): the collection of isssued async tasks 

    Returns:
    - (Chroma): the embedding vectorstore
    """
    # Wait for all scheduled tasks to complete
    if async_tasks:
        in_progress_tasks = len(async_tasks) - len(completed_tasks)
        logging.info(f"Waiting for {in_progress_tasks} async tasks to finish ...")
        await asyncio.gather(*async_tasks)
        logging.info(f"All {len(async_tasks)} async tasks finished.")

    # Save the Chroma database after processing all chunks
    if docs_db is not None:
        logging.info("Saving the vectorstore ...")
        docs_db.persist()

    return docs_db


async def process_chunks(docs_db, documents, embedding, collection_name, persist_directory, async_tasks, completed_tasks, task_id) -> Chroma:
    """
    Processes the specified chunk of (Documents).

    Parameters:
    - docs_db (Chroma): if the specified vectorstore is None, - syncronously creates one with the specified documents;
                        otherwise, - asyncronously adds the specified documents to the vectorstore. 
    - documents (List[Document]): The unstructured document splits
    - embedding: The LLM used as embedding to process documents
    - collection_name (str): the vectorstore collection name
    - persist_directory (str): The optional file path to store the embedding database; 
                                if it is not specified, (Chroma) is not persisted.
    - async_tasks (List): the collection of isssued async tasks
    - completed_tasks (List): the collection of finished async tasks
    - task_id (int): the async task id

    Returns:
    - (Chroma): the embedding vectorstore
    """
    if not docs_db:
        logging.info(f"Creating the embedding vectorstore with {embedding} for {len(documents)} document splits ...")    
        if collection_name is None:
            collection_name = DEFAULT_COLLECTION_NAME
        docs_db = Chroma.from_documents(
            documents=documents,
            collection_name=collection_name,
            embedding=embedding, 
            persist_directory=persist_directory
        )
        logging.info("Finished the creation of embedding vectorstore.")    
    else:
        logging.info(f"The async task #{task_id} is updating the embedding vectorstore with {len(documents)} document splits ...")
        async_task = asyncio.create_task(docs_db.aadd_documents(documents=documents))
        async_task.add_done_callback(lambda future, i=task_id: on_task_completion(future, i, completed_tasks))       
        async_tasks.append(async_task)

    return docs_db


def adjust_batch_size(batch_size, items_count):
    max_thread = items_count / batch_size
    # Limit to 10 concurrent thread (the first batch is processed synchronously)
    if max_thread > 11:
        batch_size = items_count / 11

    return batch_size    


async def process_splits_in_chunks(embedding, documents, chunk_size, collection_name, persist_directory) -> Chroma:
    """
    Add the specified (Documents) in chunks to a new (Chroma) vectorstore.

    Parameters:
    - embedding: The LLM used as embedding to process documents
    - documents (List[Document]): The unstructured document splits
    - chunk_size (int): The size of each batch/chunk added to a vectorstore
    - collection_name (str): the vectorstore collection name
    - persist_directory (str): The optional file path to store the embedding database; 
                                if it is not specified, (Chroma) is not persisted.

    Returns:
    - (Chroma): the embedding vectorstore
    """
    docs_db = None
    total_count = len(documents)
    async_tasks = []
    completed_tasks = []

    documents_count = len(documents)
    chunk_size = adjust_batch_size(batch_size=chunk_size, items_count=documents_count)

    # Process in chunks
    for i in range(0, documents_count, chunk_size):
        document_chunk = documents[i:i + chunk_size]
        batch_id = f"{i // chunk_size + 1}/{total_count // chunk_size + 1}"
        logging.info(f"Processing the batch {batch_id}: {len(document_chunk)} documents")

        if document_chunk:
            docs_db = await process_chunks(docs_db, document_chunk, embedding, collection_name, persist_directory, async_tasks, completed_tasks, i)

    return await wait_for_tasks(docs_db, async_tasks, completed_tasks)


async def process_files_in_chunks(embedding, file_paths, chunk_size, collection_name, persist_directory) -> Chroma:
    """
    Add the specified (Documents) in chunks to a new (Chroma) vectorstore.

    Parameters:
    - embedding: The LLM used as embedding to process documents
    - file_paths (List[str]): Files with JSON storing unstructured document splits
    - chunk_size (int): The size of each batch/chunk added to a vectorstore
    - collection_name (str): the vectorstore collection name
    - persist_directory (str): The optional file path to store the embedding database; 
                                if it is not specified, (Chroma) is not persisted.

    Returns:
    - (Chroma): the embedding vectorstore
    """
    docs_db = None
    total_count = len(file_paths)
    async_tasks = []
    completed_tasks = []
    # Process in chunks
    task_id = 0

    files_count = len(file_paths)
    chunk_size = adjust_batch_size(batch_size=chunk_size, items_count=files_count)

    for i in range(0, files_count, chunk_size):
        file_chunk = file_paths[i:i + chunk_size]
        batch_id = f"{i // chunk_size + 1}/{total_count // chunk_size + 1}"
        logging.info(f"Processing the batch {batch_id}: {len(file_chunk)} documents")
        if file_chunk:
            documents = [load_document_split(open(file_path, 'r')) for file_path in file_chunk if file_path]
            logging.info(f"Starting the task {task_id} ...")
            docs_db = await process_chunks(docs_db, documents, embedding, collection_name, persist_directory, async_tasks, completed_tasks, task_id)
            task_id = task_id + 1

    return await wait_for_tasks(docs_db, async_tasks, completed_tasks)


async def create_embedding_database(documents, model_name, chunk_size, collection_name, persist_directory) -> Chroma:
    """
    Creates a (Chroma) embedding vectorstore which stores processed unstructured document splits.

    Parameters:
    - documents (List[Document]): The unstructured document splits
    - model_name (str): The embedding model name
    - chunk_size (int): The size of each batch/chunk added to a vectorstore
    - collection_name (str): the vectorstore collection name
    - persist_directory (str): The optional file path to store the embedding vectorstore; 
                               if it is not specified, (Chroma) is not persisted.

    Returns:
    - (Chroma): the embedding vectorstore
    """
    if documents is None or len(documents) == 0:
        logging.warning(f"Cannot create an embedding database from empty list of documents!")  
        return None  

    embedding = ModelInfo.create_embedding(model_name=model_name)

    return await process_splits_in_chunks(
        embedding=embedding, 
        documents=documents, 
        chunk_size=chunk_size,
        collection_name=collection_name,
        persist_directory=persist_directory
    )


async def create_embedding_database_from_zip(zip_file, model_name, chunk_size, collection_name, persist_directory) -> Chroma:
    """
    Creates a (Chroma) embedding vectorstore from the spcified zip file which stores processed unstructured document splits.

    Parameters:
    - zip_file (str): The full path to the zip file
    - model_name (str): The embedding model name
    - chunk_size (int): The size of each batch/chunk added to a vectorstore
    - collection_name (str): the vectorstore collection name
    - persist_directory (str): The optional file path to store the embedding vectorstore; 
                               if it is not specified, (Chroma) is not persisted.

    Returns:
    - (Chroma): the embedding vectorstore
    """
    unzip_folder = load_zip_with_splits(zip_file=zip_file) 

    if unzip_folder is None:
        logging.warning(f"Cannot create an embedding database from empty zip: {zip_file}")  
        return None  
    
    # Collect all file paths
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(unzip_folder):
        for file_name in filenames:
            full_path = os.path.join(dirpath, file_name)
            file_paths.append(full_path)

    embedding = ModelInfo.create_embedding(model_name=model_name)

    return await process_files_in_chunks(
        embedding=embedding, 
        file_paths=file_paths, 
        chunk_size=chunk_size,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

def create_vector_store(dir_path, file_loader_query, model_name, collection_name, persist_directory) -> Chroma:
    """
    Creates a (Chroma) embedding vectorstore which stores processed unstructured document splits
    associates with the specified file types.

    Parameters:
    - dir_path (str): The root directory where the search for documents is performed
    - file_loader_query (FileType): The file type to load
    - model_name (str): The embedding model name; if it is not specified, it is set to "BAAI/bge-small-en"
    - collection_name (str): the vectorstore collection name
    - persist_directory (str): The optional file path to store the embedding vectorstore; 
                               if it is not specified, (Chroma) is not persisted.

    Returns:
    - (Chroma): the embedding vectorstore if documents are found and processed; otherwise - None.
    """
    # Load documents
    split_docs = load_documents(dir_path, file_loader_query)

    if split_docs:
        # Create embeddings and database
        return asyncio.run(create_embedding_database(
            documents=split_docs, 
            model_name=model_name, 
            chunk_size=BATCH_SIZE,
            collection_name=collection_name,
            persist_directory=persist_directory
        ))
    
    return None

def load_vector_store(model_name, collection_name, persist_directory) -> Chroma:
    """
    Load the Chroma for the vectorstore persisted in the specified directory.

    Parameters:
    - model_name (str): The embedding model name
    - collection_name (str): the vectorstore collection name
    - persist_directory (str): The optional file path to store the embedding vectorstore; 
                               if it is not specified, (Chroma) is not persisted.

    Returns:
    - (Chroma): the embedding vectorstore if documents are found and processed; otherwise - None.
    """
    logging.info(f"Creating (HuggingFaceInstructEmbeddings) for '{model_name}' ...")
    
    embedding = ModelInfo.create_embedding(model_name=model_name)

    if collection_name is None:
        collection_name = DEFAULT_COLLECTION_NAME

    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding,
        client_settings=CHROMA_SETTINGS,
    )


if __name__ == "__main__":      
    """
    Main function to load documents, split them, and create a vectorstore with this splits of found documents.
    """
    # Set the logging level to INFO
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create the parser
    parser = argparse.ArgumentParser(description="Creating the embedding database.")

    # Add the arguments
    parser.add_argument(
        '--dir_path', 
        type=str, help='Root directory where to look for documents.', 
        default="."
    )
    parser.add_argument(
        '--zip_file', 
        type=str, 
        help='(Optional) The zip file contains unstructured document splits. If this parameter is specified, --dir_path, --file_types, --file_patterns are ignored.', 
        default=None
    )
    parser.add_argument(
        '--file_types', 
        type=str, nargs='+', 
        help='List of file extensions without the do: md; java; xml; html; pdf'
    )
    parser.add_argument(
        '--file_patterns', 
        nargs='+', 
        help='Name patterns for each file type; for example: --file_patterns "java:**/*Function* html:**/*"', 
        default=[]
    )   
    parser.add_argument(
        '--persist_directory', 
        type=str, 
        help='(Optional) The path to the directory where the vectorsstore will be persisted.', 
        default=None
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        help='The name of embedding model for analyzing unstructured document splits.', 
        default=None
    )
    parser.add_argument(
        '--collection_name', 
        type=str, 
        help='The name of vectorsstore.', 
        default=None
    )
    parser.add_argument(
        '--test_question', 
        type=str, 
        help='(Optional) Indicates if the created vectorsstore should be tested.', 
        default=None
    )
      
    # Parse the arguments
    args = parser.parse_args()

    logging.info(f"Creating the vectorsstore the arguments: {args}")
    
    # Call the create_vector_store function
    start_time = time.time()
    
    if args.zip_file:
        docs_db = asyncio.run(create_embedding_database_from_zip(
            zip_file=args.zip_file, 
            model_name=args.model_name,
            chunk_size=BATCH_SIZE,
            collection_name=args.collection_name,
            persist_directory=args.persist_directory
        ))
    else:
        file_loader_query = get_FileLoaderQuery(args.file_types, args.file_patterns)   
        docs_db = create_vector_store(
            dir_path=args.dir_path, 
            file_loader_query=file_loader_query, 
            model_name=args.model_name,
            collection_name=args.collection_name,
            persist_directory=args.persist_directory
        )

    create_manifest(collection_name=args.collection_name, model_name=args.model_name, persist_directory=args.persist_directory)

    elapsed_time_msg = get_elapse_time_message(start_time=start_time)

    logging.info(f"Finished the creation of vectorsstore creation in {elapsed_time_msg}.")

    # Test a new vectorstore
    doc_ids = docs_db.get()["ids"]
    logging.info(f"The vectorestore stores {len(doc_ids)} documents")
    
    if args.test_question is not None:
        retriever = docs_db.as_retriever()
        answer = retriever.invoke(args.test_question)
        logging.info(f"ANSWER: {answer[0].page_content}")

    # Optionally, you can print or log the result
    logging.info(f"The embedding database: {docs_db}")