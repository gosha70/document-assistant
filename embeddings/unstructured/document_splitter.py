# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import os
from typing import List

from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document

from .base_file_converter import BaseFileConverter
from .file_type import FileType
from .pdf_converter import PdfConverter
from .generic_converter import GenericConverter
from .csv_converter import CsvConverter
from .excel_converter import ExcelConverter
from .markdown_converter import MarkdownConverter
from .html_converter import HtmlConverter
from .java_converter import JavaConverter
from .json_converter import JsonConverter
from .js_converter import JsConverter
from .python_converter import PythonConverter     
from .rtf_converter import RtfConverter
from .xml_converter import XmlConverter

class DocumentSplitter:
    """
    Finds files and splits them into unstructured text.
    """
    def __init__(self, logging):
        self.logging = logging
        self.converters = {
            FileType.PDF: PdfConverter(logging=logging),
            FileType.CSV: CsvConverter(logging=logging),
            FileType.DDL: GenericConverter(file_type=FileType.DDL, language=None, logging=logging),
            FileType.SQL: GenericConverter(file_type=FileType.SQL, language=None, logging=logging),
            FileType.EXCEL: ExcelConverter(logging=logging),
            FileType.JAVA: JavaConverter(logging=logging),
            FileType.JS: JsConverter(logging=logging),
            FileType.JSON: JsonConverter(logging=logging),
            FileType.HTML: HtmlConverter(logging=logging),
            FileType.MARKDOWN: MarkdownConverter(logging=logging),
            FileType.PYTHON: PythonConverter(logging=logging),
            FileType.RICH_TEXT: RtfConverter(logging=logging),
            FileType.TEXT: GenericConverter(file_type=FileType.TEXT, language=None, logging=logging),
            FileType.XML: XmlConverter(logging=logging),
            FileType.XSL: GenericConverter(file_type=FileType.XSL, language=None, logging=logging),
            FileType.YAML: GenericConverter(file_type=FileType.YAML, language=None, logging=logging)
        }
     
    def get_converter(self, file_type: FileType):
        """Gets the BaseFileConverter for a given FileType."""
        return self.converters.get(file_type)
    
    def find_files(self, dir_path: str, file_extension: str) -> List[str]:
        """
        Finds sources corresponding the specified file type.

        Parameters:
        - dir_path (str): The root directory where the search for documents is performed
        - file_type (str): The optional pattern for a file name
        Returns:
        - (List[str]): files found in the specified directory
        """
        self.logging.info(f"Loading {file_extension} ...")   
        found_files = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(file_extension):
                    found_files.append(os.path.join(root, file))

        self.logging.info(f"Loaded {len(found_files)} files.")
        return found_files
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        Processes a single file at th runtime

        Parameters:
        - file_path (str): The path to file 
        Returns:
        - (List[Document]): the list of unstructured Documents
        """ 
        file_type = FileType.get_file_type_by_extension(file_name=file_path)   
        test_splitter = BaseFileConverter.get_text_splitter(file_type=file_type) 
        return self.load_split_file(text_splitter=test_splitter, file_type=file_type, file_path=file_path)  
    
    def load_split_file(self, text_splitter: TextSplitter, file_type: FileType, file_path: str) -> List[Document]:
        """
        Reads and processes a single file

        Parameters:
        - text_splitter (TextSplitter): The text splitter
        - file_type (FileType): The file type to load
        - file_path (str): The path to file
        Returns:
        - (List[Document]): the list of unstructured PDF content
        """         
        converter = self.get_converter(file_type)    
        return converter.load_and_split_file(text_splitter=text_splitter, file_path=file_path)

    def load_and_split(self, dir_path: str, text_splitter: TextSplitter, file_type: FileType, file_pattern: str = None) -> List[Document]:   
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
        converter = self.get_converter(file_type)
        # Ensure the directory path ends with '/'
        if not dir_path.endswith('/'):
            dir_path += '/'
        
        file_name = file_pattern if file_pattern is not None else "**/*"
        self.logging.info(f"Loading {file_type.get_extension()} with names confirming the name pattern: '{file_name}'")         
        documents = converter.load_and_split(dir_path=dir_path, file_pattern=file_pattern)                
        self.logging.info(f"Loaded {len(documents)} {file_type.get_extension()} documents")
        return text_splitter.split_documents(documents)
    

    