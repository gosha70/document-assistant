import os
import argparse
import logging
import datetime
import json
from collections import defaultdict
from enum import Enum
from typing import List
from langchain_core.documents import Document
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import (
    Language, 
    RecursiveCharacterTextSplitter,
    TextSplitter
)
from langchain.document_loaders import PyPDFLoader

from .embeddings_constants import CHUNK_SIZE, CHUNK_OVERLAP

"""
Enum defines supported file types/extensions
"""
class FileType(str, Enum): 
    JAVA = "java"
    HTML = "html"
    MARKDOWN = "md"
    PDF = "pdf"
    XML = "xml"

    def get_file_type(value):
        try:
            return FileType(value)
        except ValueError:
            return None

    def get_extension(self) -> str:
        return f".{self.value}"

file_type_per_language = {
    FileType.JAVA: Language.JAVA, 
    FileType.HTML: Language.HTML, 
    FileType.MARKDOWN: Language.MARKDOWN
}

class FileLoaderQuery:
    def __init__(self):
        self.patterns = {}

    def add_file_type(self, file_type, patterns):
        """Add a file type with associated patterns."""
        if file_type in self.patterns:
            self.patterns[file_type].update(patterns)
        else:
            self.patterns[file_type] = set(patterns)

    def get_patterns(self, file_type):
        """Retrieve patterns for a specific file type."""
        return self.patterns.get(file_type, set())

    def remove_file_type(self, file_type):
        """Remove a file type and its patterns."""
        if file_type in self.patterns:
            del self.patterns[file_type]

    def __str__(self):
        return str(self.patterns)

"""
Loads and processes all PDF files in the specified directory

Parameters:
- dir_path (str): The root directory where the search for PDF is performed
Returns:
- (List[str]): unstructured PDF splits
"""
def read_pdf(dir_path) -> List[str]:
    loader = PyPDFLoader(dir_path)
    return loader.load_and_split()

"""
Finds sources corresponding the specified file type.

Parameters:
- dir_path (str): The root directory where the search for documents is performed
- file_type ({str}): The optional pattern for a file name
Returns:
- (List[str]): file
"""
def find_files(dir_path, file_extension) -> List[str]:
    logging.info(f"Loading {file_extension} ...")   
    found_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(file_extension):
                found_files.append(os.path.join(root, file))

    logging.info(f"Loaded {len(found_files)} files.")
    return found_files               

"""
Loads PDF files in the specified directory.

Parameters:
- dir_path (str): The root directory where the search for documents is performed

Returns:
- (List[str]): unstructured document splits
""" 
def load_and_split_pdf(dir_path) -> List[Document]:
    split_docs = []
    files = find_files(dir_path, '.pdf')
    for file_path in files:
        document = read_pdf(file_path)
        logging.info(f"PDF splits: {len(document)}") 
        split_docs.extend(document)
    
    return split_docs
    
"""
Finds files corresponding the specified file type and optional pattern name; 
then loads them into unstructured documents; at the end, 
split them via the specified (TextSplitter).

Parameters:
- dir_path (str): The root directory where the search for documents is performed
- text_splitter (TextSplitter): The text splitter
- file_type (FileType): The file type to load
- file_pattern (str): The optional pattern for a file name

Returns:
- (List[Document]): unstructured document splits
"""
def load_and_split(dir_path, text_splitter, file_type, file_pattern = None) -> List[Document]:

    file_name = file_pattern if file_pattern is not None else "**/*"

    logging.info(f"Loading  {file_type.get_extension()} with names confirming the name pattern: '{file_name}'") 
    loader = GenericLoader.from_filesystem(
        dir_path,
        glob=file_name,
        suffixes=[file_type.get_extension()],
        parser=LanguageParser(),
    )
    documents = loader.load()    
    logging.info(f"Loaded {len(documents)} {file_type.get_extension()} documents")   

    return text_splitter.split_documents(documents)

"""
Returns (TextSplitter) for the specified language.

Parameters:
- file_type (FileType): The file type enum indicating which (TextSplitter) to use

Returns:
- (TextSplitter)

See: https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.Language.html
"""
def get_text_splitter(file_type) -> TextSplitter:
    language = file_type_per_language.get(file_type)
    if language is not None:
        text_separators = RecursiveCharacterTextSplitter.get_separators_for_language(language)
    else:    
        text_separators = None

    return RecursiveCharacterTextSplitter(
        separators = text_separators,
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        keep_separator=True
    )    

"""
Loads files in the specified directory into unstructured document splits.

Parameters:
- dir_path (str): The root directory where the search for documents is performed
- file_loader_query (FileLoaderQuery): The FileLoaderQuery holds the search criteria for files to laod and analyze

Returns:
- (List[str]): unstructured document splits

See: https://api.python.langchain.com/en/v0.0.345/documents/langchain_core.documents.base.Document.html
"""
def load_documents(dir_path, file_loader_query) -> List[Document]:
    try:
        split_docs = []
        # Iterate over the file types and their patterns
        for file_type in file_loader_query.patterns:
            text_splitter = get_text_splitter(file_type)
            if text_splitter is None:
                logging.warning(f"Cannot find (TextSplitter) for {file_type.get_extension()}")
                continue
            patterns = file_loader_query.get_patterns(file_type)
            for pattern in patterns:
                if file_type == FileType.PDF:
                    split_docs = load_and_split_pdf(dir_path)
                else:     
                    if file_type is None:
                        raise ValueError(f"Got the unsupported field type '{file_type.get_extension()}'")
                    split_docs = load_and_split(dir_path, text_splitter, file_type, pattern)
                split_docs.extend(split_docs)

        logging.info(f"Total number of unstructured document splits: {len(split_docs)}")

        return split_docs 
    except Exception as error:
        logging.error(f"Failed to process documents with extensions '{file_loader_query}' found in the path '{dir_path}': {str(error)}", exc_info=True)

        return None

"""
Saves each split document as a separate file in the specified output directory.
If output_dir is None, creates a new directory in the current directory.

Parameters:
- split_docs (List[Document]): List of split document objects
- output_dir (str): The directory where the split documents will be saved. Defaults to None.

Returns: the directory where doocuments are saved
"""
def save_documents_to_disk(split_docs, output_dir=None):  
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
    
    
def get_FileLoaderQuery(file_types, file_patterns) -> FileLoaderQuery:
    # Create a mapping from file type to patterns
    pattern_mapping = defaultdict(list)
    for pattern in file_patterns:
        file_type, file_pattern = pattern.split(':', 1)
        pattern_mapping[file_type].append(file_pattern)

    # Create an instance of FileTypePatterns and add file types and patterns
    file_loader_query = FileLoaderQuery()
       
    for file_type_name in file_types: 
        file_type = FileType.get_file_type(file_type_name)
        if file_type is not None:
            patterns = pattern_mapping.get(file_type, ['**/*'])  # Default pattern if not specified      
            file_loader_query.add_file_type(file_type, set(patterns))
        else:
            logging.error(f"Unsupported file type: {file_type_name}")  

    return file_loader_query         
    
"""
Main function to load documents, split them, and save the splits to disk.
"""
def main():
    # Set the logging level to INFO
    logging.basicConfig(level=logging.INFO)

    # Create the parser
    parser = argparse.ArgumentParser(description="Creating the embedding database.")

    # Add the arguments
    parser.add_argument('--dir_path', type=str, help='Root directory where to look for documents.', default=".")
    parser.add_argument('--file_types', type=str, nargs='+', help='List of file extensions without the do: md; java; xml; html; pdf')
    parser.add_argument('--file_patterns', nargs='+', help='Name patterns for each file type; for example: --file_patterns "java:**/*Function* html:**/*"', default=[])   
    parser.add_argument('--persist_directory', type=str, help='(Optional) The path to the directory where unstructured document splits are saved.', default=None)
  
    # Parse the arguments
    args = parser.parse_args()

    logging.info(f"Searching and processing documents with the arguments: {args}")

    file_loader_query = get_FileLoaderQuery(args.file_types, args.file_patterns)

    # Load and split documents
    split_docs = load_documents(args.dir_path, file_loader_query)

    if split_docs is not None:
        # Save split documents to disk
        doc_dir = save_documents_to_disk(split_docs, args.persist_directory)
        logging.info(f"{len(split_docs)} documet chunks are saved in {doc_dir}.")
    else:
        logging.error("No documents were loaded or split.")

if __name__ == "__main__":
    main() 

