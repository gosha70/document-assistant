import os
import time
import argparse
import logging
from datetime import datetime
import json
import zipfile
import tempfile
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
from langchain.document_loaders import (
    JSONLoader, 
    PythonLoader,
    PyPDFLoader, 
    TextLoader,  
    UnstructuredCSVLoader,  
    UnstructuredHTMLLoader, 
    UnstructuredMarkdownLoader, 
    UnstructuredRTFLoader,
    UnstructuredXMLLoader,
    UnstructuredExcelLoader
)

from .embeddings_constants import CHUNK_SIZE, CHUNK_OVERLAP, get_elapse_time_message, KWARGS_PARAM_NAME, PAGE_CONTENT_PARAM_NAME, METADATA_PARAM_NAME

class FileType(str, Enum): 
    """
    Enum defines supported file types/extensions
    """
    CSV = "csv"
    DDL = "ddl"
    EXCEL = "xlsx"
    JAVA = "java"
    JS = "js"
    JSON  = "json"
    HTML = "html"
    MARKDOWN = "md"
    PDF = "pdf"
    PYTHON = "py"
    RICH_TEXT = "rtf"
    SQL = "sql"
    TEXT = "txt"
    XML = "xml"
    XSL = "xsl"
    YAML = "yaml"

    def get_file_type(value):
        try:
            return FileType(value)
        except ValueError:
            return None

    def get_extension(self) -> str:
        return f".{self.value}"

file_type_per_language = {
    FileType.JAVA: Language.JAVA, 
    FileType.JS: Language.JS,
    FileType.HTML: Language.HTML, 
    FileType.MARKDOWN: Language.MARKDOWN,
    FileType.PYTHON: Language.PYTHON,
    FileType.SQL: Language.SOL,
    FileType.DDL: Language.SOL
}

class FileLoaderQuery:
    """
    Holds the configuration for ssearching files to analyze
    """
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

def find_files(dir_path, file_extension) -> List[str]:
    """
    Finds sources corresponding the specified file type.

    Parameters:
    - dir_path (str): The root directory where the search for documents is performed
    - file_type (str): The optional pattern for a file name
    Returns:
    - (List[str]): files found in the specified directory
    """
    logging.info(f"Loading {file_extension} ...")   
    found_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(file_extension):
                found_files.append(os.path.join(root, file))

    logging.info(f"Loaded {len(found_files)} files.")
    return found_files           

def read_pdf(file_path) -> List[str]:
    """
    Reads and processes a single PDF file

    Parameters:
    - file_path (str): The path to PDF
    Returns:
    - (List[str]): unstructured PDF splits
    """
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def load_and_split_pdf(dir_path) -> List[Document]:
    """
    Loads PDF files in the specified directory.

    Parameters:
    - dir_path (str): The root directory where the search for documents is performed

    Returns:
    - (List[str]): unstructured document splits
    """ 
    split_docs = []
    files = find_files(dir_path, '.pdf')
    for file_path in files:
        document = read_pdf(file_path)
        logging.info(f"PDF splits: {len(document)}") 
        split_docs.extend(document)
    
    return split_docs
 
def load_and_split(dir_path, text_splitter, file_type, file_pattern = None) -> List[Document]:   
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

def get_text_splitter(file_type) -> TextSplitter:
    """
    Returns (TextSplitter) for the specified language.

    Parameters:
    - file_type (FileType): The file type enum indicating which (TextSplitter) to use

    Returns:
    - (TextSplitter)

    See: https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.Language.html
    """
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

def load_split_file(text_splitter, file_type, file) -> List[Document]:
    if file_type == FileType.PDF:
        return read_pdf(file)
    else:    
        if file_type == FileType.CSV:
            loader = UnstructuredCSVLoader(file_path=file)
        elif file_type == FileType.HTML:
            loader = UnstructuredHTMLLoader(file_path=file)
        elif file_type == FileType.JSON:
            loader = JSONLoader(file_path=file, jq_schema='.', text_content=False)
        elif file_type == FileType.MARKDOWN:
            loader = UnstructuredMarkdownLoader(file_path=file)
        elif file_type == FileType.XML:
            loader = UnstructuredXMLLoader(file_path=file)
        elif file_type == FileType.EXCEL:
            loader = UnstructuredExcelLoader(file_path=file)
        elif file_type == FileType.PYTHON:
            loader = PythonLoader(file_path=file)
        elif file_type == FileType.RICH_TEXT:
            loader = UnstructuredRTFLoader(file_path=file)           
        else:
            loader = TextLoader(file_path=file)    

        try: 
          return loader.load_and_split(text_splitter=text_splitter)
        except Exception as error:
          logging.error(f"Failed to process '{file}' with the text splitter '{text_splitter}'; the file will be procesed with TextLoader instead: {str(error)}", exc_info=False)

        return TextLoader(file_path=file).load_and_split(text_splitter=text_splitter) 
        

def load_supported_documents(dir_path) -> List[Document]:
    """
    Finds and loads all files corresponding to supported file types and counts them.

    Parameters:
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
            text_splitter = get_text_splitter(file_type)
            if text_splitter is None:
                logging.warning(f"Cannot find (TextSplitter) for {file_type.get_extension()}")
                continue
            for file_path in files:
                file_splits = load_split_file(text_splitter, file_type, file_path)
                split_docs.extend(file_splits)
                file_type_counts[file_type] += 1

    logging.info(f"Total document splits: {len(split_docs)}")  
    # Log the count of each file type found
    for file_type, count in file_type_counts.items():
        if count > 0:
            logging.info(f"Found {count} '{file_type.value}' files.")
  
    return split_docs

def load_documents(dir_path, file_loader_query) -> List[Document]:
    """
    Loads files in the specified directory into unstructured document splits.

    Parameters:
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

def save_splits_to_disk(split_docs, output_dir=None):  
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

    
def get_FileLoaderQuery(file_types, file_patterns) -> FileLoaderQuery:
    # Create a mapping from file type to patterns
    # Create a mapping from file type to patterns
    pattern_mapping = {}
    for pattern in file_patterns:
        file_type, file_pattern = pattern.split(':', 1)
        if file_type not in pattern_mapping or pattern_mapping[file_type] is None:
            pattern_mapping[file_type] = set()
        pattern_mapping[file_type].add(file_pattern)

    # Create an instance of FileTypePatterns and add file types and patterns
    file_loader_query = FileLoaderQuery()

    for file_type_name in file_types: 
        file_type = FileType.get_file_type(file_type_name)
        if file_type is not None:
            patterns = pattern_mapping.get(file_type, ['**/*'])  # Default pattern if not specified      
            file_loader_query.add_file_type(file_type, patterns)
        else:
            logging.error(f"Unsupported file type: {file_type_name}")  

    return file_loader_query         

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
    # Load and split documents
    start_time = time.time()
    if args.file_types is None:
        split_docs = load_supported_documents(args.dir_path)
    else:
        file_loader_query = get_FileLoaderQuery(args.file_types, args.file_patterns)    
        split_docs = load_documents(args.dir_path, file_loader_query)

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