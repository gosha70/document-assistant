# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    Language, 
    RecursiveCharacterTextSplitter,
    TextSplitter
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from embeddings.embeddings_constants import CHUNK_SIZE, CHUNK_OVERLAP
from .file_type import FileType

file_type_per_language = {
    FileType.JAVA: Language.JAVA, 
    FileType.JS: Language.JS,
    FileType.HTML: Language.HTML, 
    FileType.MARKDOWN: Language.MARKDOWN,
    FileType.PYTHON: Language.PYTHON,
    FileType.SQL: Language.SOL,
    FileType.DDL: Language.SOL
}

class BaseFileConverter(ABC):
    """Converts `File` to Documents"""
  
    def __init__(self, file_type: FileType, language: Optional[Language] =None, logging=None):
        """Initializes BaseFileConverter with optional Language and Logging."""
        self.file_type = file_type
        self.language = language
        self.logging = logging

    @staticmethod
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
   
    def get_language(self) -> Language:
        """
        Gets Language associated with this Unstructured API

        Returns:
        - Language: the language
        """
        return self.language
    
    def log_info(self, messsage: str):
        if self.logging is None: 
            print(messsage)
        else:
            self.logging.info(messsage)
       
    def load_and_split_file(self, text_splitter: TextSplitter, file_path: str) -> List[Document]:
        """
        Reads and processes a single file

        Parameters:
        - text_splitter (TextSplitter): The text splitter
        - file_path (str): The path to file
        Returns:
        - (List[Document]): the list of unstructured PDF content
        """
        return TextLoader(file_path=file_path).load_and_split(text_splitter=text_splitter)         

    def load_and_split_files(self, dir_path: str, file_pattern: str) -> List[Document]:
        """
        Reads and processes files which match the specified pattern

        Parameters:
        - dir_path (str): The root directory where the search for documents is performed
        - file_pattern (str): The optional pattern for a file name

        Returns:
        - (List[Document]): the list of unstructured content
        """
        
        if self.language is None:
            language_parser = LanguageParser(parser_threshold=2000)
        else:
            language_parser = LanguageParser(language=self.language, parser_threshold=1000)

        loader = GenericLoader.from_filesystem(
            path=dir_path,
            glob=file_pattern,
            suffixes=[self.file_type.get_extension()],
            parser=language_parser,
        )
        
        return loader.load()     

    

        