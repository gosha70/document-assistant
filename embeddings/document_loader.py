# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import os
import time
import argparse
import logging
from datetime import datetime
import json
import zipfile
import tempfile
from typing import List
from langchain_core.documents import Document

from .embeddings_constants import get_elapse_time_message, KWARGS_PARAM_NAME, PAGE_CONTENT_PARAM_NAME, METADATA_PARAM_NAME
from embeddings.unstructured.file_type import FileType
from embeddings.unstructured.file_loader_query import FileLoaderQuery
from embeddings.unstructured.document_splitter import DocumentSplitter
from embeddings.unstructured.base_file_converter import BaseFileConverter


def load_supported_documents(document_splitter: DocumentSplitter, dir_path: str) -> List[Document]:
    """
    Finds and loads all files corresponding to supported file types and counts them.

    Parameters:
    - document_splitter (DocumentSplitter): helps to find and split files/documents
    - dir_path (str): The root directory where the search for documents is performed

    Returns:
    - (List[Document]): unstructured document splits
    """
    logging.info("Loading files with supported extensions...")   
    files_by_type = {file_type: [] for file_type in FileType} 
    for root, _, files in os.walk(dir_path):
        for file in files:
            for file_type in FileType:
                if file.endswith(file_type.get_extension()):
                    file_path = os.path.join(root, file)
                    files_by_type[file_type].append(file_path)

    split_docs = [] 
    file_type_counts = {file_type: 0 for file_type in FileType} 
    for file_type, files in files_by_type.items():
        if len(files) > 0: 
            text_splitter = BaseFileConverter.get_text_splitter(file_type)
            if text_splitter is None:
                logging.warning(f"Cannot find (TextSplitter) for {file_type.get_extension()}")
                continue
            for file_path in files:
                file_splits = document_splitter.load_split_file(text_splitter, file_type, file_path)
                split_docs.extend(file_splits)
                file_type_counts[file_type] += 1

    logging.info(f"Total document splits: {len(split_docs)}")  
    # Log the count of each file type found
    for file_type, count in file_type_counts.items():
        if count > 0:
            logging.info(f"Found {count} '{file_type.value}' files.")
  
    return split_docs

def load_documents(document_splitter: DocumentSplitter, dir_path: str, file_loader_query: FileLoaderQuery) -> List[Document]:
    """
    Loads files in the specified directory into unstructured document splits.

    Parameters:
    - document_splitter (DocumentSplitter): helps to find and split files/documents
    - dir_path (str): The root directory where the search for documents is performed
    - file_loader_query (FileLoaderQuery): The FileLoaderQuery holds the search criteria for files to laod and analyze

    Returns:
    - (List[Document]): unstructured document splits

    See: https://api.python.langchain.com/en/v0.0.345/documents/langchain_core.documents.base.Document.html
    """
    try:
        split_docs = []
        # Iterate over the file types and their patterns
        for file_type in file_loader_query.patterns:
            text_splitter = BaseFileConverter.get_text_splitter(file_type)
            if text_splitter is None:
                logging.warning(f"Cannot find (TextSplitter) for {file_type.get_extension()}")
                continue
            patterns = file_loader_query.get_patterns(file_type)
            for pattern in patterns:
                if file_type is None:
                    raise ValueError(f"Got the unsupported field type '{file_type.get_extension()}'")
                file_type_docs = document_splitter.load_and_split(dir_path, text_splitter, file_type, pattern)                    
                split_docs.extend(file_type_docs)

        logging.info(f"Total number of unstructured document splits: {len(split_docs)}")

        return split_docs 
    except Exception as error:
        logging.error(f"Failed to process documents with extensions '{file_loader_query}' found in the path '{dir_path}': {str(error)}", exc_info=True)

        return None

def save_splits_to_disk(split_docs, output_dir: str = None):  
    """
    Saves each split document as a separate file in the specified output directory.
    If output_dir is None, creates a new directory in the current directory.

    Parameters:
    - split_docs (List[Document]): List of split document objects
    - output_dir (str): The directory where the split documents will be saved. Defaults to None.

    Returns: the directory where doocuments are saved
    """
    if output_dir is None:
        # Create a new directory with a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), f"split_docs_{timestamp}")
     
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, doc in enumerate(split_docs):
        doc_json = doc.to_json()  # Convert Document object to Json
        with open(os.path.join(output_dir, f"doc_split_{i}.json"), "w", encoding="utf-8") as file:
            json.dump(doc_json, file, indent=4)

    return output_dir 

def load_document_split(split_file) -> Document:
    """
    Craete Document from the specified JSON file.

    Parameters:
    - split_file (File): the JSON file sstoring a single unstructured document split

    Returns (Document)
    """
    try:
        # Read and process the content as JSON
        json_content = split_file.read()

        # Parse the JSON content
        data = json.loads(json_content)

        # Access the "page_content" field
        page_content = data[KWARGS_PARAM_NAME][PAGE_CONTENT_PARAM_NAME]

        # Access the "metadata" field
        metadata = data[KWARGS_PARAM_NAME][METADATA_PARAM_NAME]

        # Transform the data into a langchain_core.documents.Document
        # Assuming the JSON structure fits the Document's requirements
        return Document(page_content=page_content, metadata=metadata)
    except Exception as error:
        print(f"File {split_file} is not a valid JSON: {str(error)}")

    return None

def load_zip_with_splits(zip_file, unzip_folder=None) -> str:
    """
    Extracts the specified zip with unstructured document splits to the specified folder.

    Parameters:
    - zip_file (List[Document]): List of split document objects
    - output_dir (str): The directory where the split documents will be saved. Defaults to None.

    Returns: the directory where doocuments are saved
    """    
    if unzip_folder is None:
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_folder = f"unzip_{curr_time}" 
        unzip_folder = tempfile.mkdtemp(prefix=new_folder)
    else:
        # Create the directory if it doesn't exist
        os.makedirs(unzip_folder, exist_ok=True)

    # Open the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all the contents into the directory
        zip_ref.extractall(unzip_folder)

    return unzip_folder    

def main():        
    """
    Main function to load documents, split them, and save the splits to disk.
    """
    # Set the logging level to INFO    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create the parser
    parser = argparse.ArgumentParser(description="Creating the vectorsstore ...")

    # Add the arguments
    parser.add_argument(
        '--dir_path', 
        type=str, 
        help='The root directory where to look for documents.', 
        default="."
    )
    parser.add_argument(
        '--file_types', 
        type=str, nargs='+', 
        help='The list of file extensions without the dot: md; java; xml; html; pdf'
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
        help='(Optional) The path to the directory where unstructured document splits are saved.', 
        default=None
    )
  
    # Parse the arguments
    args = parser.parse_args()

    logging.info(f"Searching and processing documents with the arguments: {args}")
    document_splitter = DocumentSplitter(logging)
    # Load and split documents
    start_time = time.time()
    if args.file_types is None:
        split_docs = load_supported_documents(document_splitter, args.dir_path)
    else:
        file_loader_query = FileLoaderQuery.get_file_loader_query(args.file_types, args.file_patterns)    
        split_docs = load_documents(document_splitter, args.dir_path, file_loader_query)

    elapsed_time_msg = get_elapse_time_message(start_time=start_time)
    logging.info(f"Finished the document loading in {elapsed_time_msg}.")

    if split_docs is not None:
        # Save split documents to disk
        doc_dir = save_splits_to_disk(split_docs, args.persist_directory)
        logging.info(f"{len(split_docs)} documet chunks are saved in {doc_dir}.")
    else:
        logging.error("No documents were loaded or split.")

if __name__ == "__main__":
    main() 