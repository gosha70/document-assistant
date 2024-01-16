# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
from enum import Enum

from langchain.text_splitter import Language

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

    @staticmethod
    def get_file_type(value):
        try:
            return FileType(value)
        except ValueError:
            return None

    def get_extension(self) -> str:
        return f".{self.value}"
    
    @staticmethod    
    def get_file_type_by_extension(file_name: str):
        """Returns the FileType for a given file name."""
        # Extract the extension from the file name
        extension = file_name.split('.')[-1]
        # Iterate through the FileType enum to find a match
        for file_type in FileType:
            if file_type.value == extension:
                return file_type
        return None
    
