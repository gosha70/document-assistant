# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import glob
from typing import List
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TextSplitter

from .base_file_converter import BaseFileConverter
from .file_type import FileType


class PdfConverter(BaseFileConverter):
    """Convert `PDF` to Documents"""
    
    def __init__(self, logging=None):
        super().__init__(file_type=FileType.PDF, language=None, logging=logging)
           
        
    def load_and_split_file(self, text_splitter: TextSplitter, file_path: str) -> List[Document]:
        """
        Reads and processes a single PDF file

        Parameters:
        Parameters:
        - text_splitter (TextSplitter): The text splitter
        - file_path (str): The path to file
        Returns:
        - (List[Document]): the list of unstructured PDF content
        """
        loader = PyPDFLoader(file_path)
        return loader.load_and_split()

    def load_and_split_files(self, dir_path: str, file_pattern: str) -> List[Document]:
        """
        Reads and processes PDF files which match the specified pattern

        Parameters:
        - dir_path (str): The root directory where the search for documents is performed
        - file_pattern (str): The optional pattern for a file name

        Returns:
        - (List[Document]): the list of unstructured content
        """
        files = glob.glob(f'{dir_path}{file_pattern}.pdf', recursive=True)
        split_docs = []
        for file_path in files:
            document = self.load_and_split_file(text_splitter=None, file_path=file_path)
            self.log_info(f"PDF splits: {len(document)}") 
            split_docs.extend(document)
        
        return split_docs

        