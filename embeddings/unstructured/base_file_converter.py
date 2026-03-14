# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    Language,
    TextSplitter
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from .file_type import FileType

# Maps FileType enum to file extension strings used by src.rag.chunking
_FILE_TYPE_TO_EXT = {
    FileType.JAVA: "java",
    FileType.JS: "js",
    FileType.HTML: "html",
    FileType.MARKDOWN: "md",
    FileType.PYTHON: "py",
    FileType.SQL: "sql",
    FileType.DDL: "ddl",
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
        Returns the config-driven, tokenizer-aware TextSplitter for the given file type.

        Delegates to src.rag.chunking.get_text_splitter() so that all ingestion
        paths (FastAPI /ingest and legacy CLI) use the same chunking strategy.
        """
        from src.rag.chunking import get_text_splitter
        ext = _FILE_TYPE_TO_EXT.get(file_type)
        return get_text_splitter(ext)    
   
    def get_language(self) -> Language:
        """
        Gets Language associated with this Unstructured API

        Returns:
        - Language: the language
        """
        return self.language
    
    def log_info(self, message: str, **kwargs):
        if self.logging is None:
            print(message)
        else:
            self.logging.info(message)
       
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

    

        