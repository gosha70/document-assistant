import os
import logging
from enum import Enum
from typing import List
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
def load_and_split_pdf(dir_path) -> List[str]:
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
- (List[str]): unstructured document splits
"""
def load_and_split(dir_path, text_splitter, file_type, file_pattern = None) -> List[str]:

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
"""
def load_documents(dir_path, file_loader_query) -> List[str]:
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

