# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
from langchain.text_splitter import Language
from .base_file_converter import BaseFileConverter
from .file_type import FileType


class JsConverter(BaseFileConverter):
    """Convert `JavaScript` to Documents"""
    
    def __init__(self, logging=None):
        super().__init__(file_type=FileType.JS, language=Language.JS, logging=logging)


        