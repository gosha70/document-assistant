# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.

from .file_type import FileType

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

    @staticmethod    
    def get_file_loader_query(file_types, file_patterns, logging):
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