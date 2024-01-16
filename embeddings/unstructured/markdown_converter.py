# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
from typing import List
from langchain_core.documents.base import Document
from langchain.text_splitter import (
    TextSplitter,
    Language
)    
from langchain.document_loaders import UnstructuredMarkdownLoader

from .base_file_converter import BaseFileConverter
from .file_type import FileType


class MarkdownConverter(BaseFileConverter):
    """Convert `Markdown` to Documents"""
    
    def __init__(self, logging=None):
        super().__init__(file_type=FileType.MARKDOWN, language=Language.MARKDOWN, logging=logging)
     
    def load_and_split_file(self, text_splitter: TextSplitter, file_path: str) -> List[Document]:
        """
        Reads and processes a single Markdown file

        Parameters:
        Parameters:
        - text_splitter (TextSplitter): The text splitter
        - file_path (str): The path to file
        Returns:
        - (List[Document]): the list of unstructured PDF content
        """
        try: 
            return UnstructuredMarkdownLoader(file_path=file_path).load_and_split(text_splitter=text_splitter)
        except Exception as error:
            self.log_info(f"Failed to process '{file_path}' with the text splitter '{text_splitter}'; the file will be procesed with TextLoader instead: {str(error)}", exc_info=False)

        return super().load_and_split_file(text_splitter=text_splitter, file_path=file_path)         


        