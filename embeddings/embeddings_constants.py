# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import time
from chromadb.config import Settings

# Char-level splits
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Parameters of JSON representing (Document)
KWARGS_PARAM_NAME = 'kwargs'
PAGE_CONTENT_PARAM_NAME = 'page_content'
METADATA_PARAM_NAME = 'metadata'

# Number of files to process at a time
BATCH_SIZE = 300

# Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

DEFAULT_COLLECTION_NAME = "EGOGE_DOCUMENTS_DB"

def get_elapse_time_message(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time > 3600:
        return f"{round(elapsed_time/60, ndigits=2)} hours"
    elif elapsed_time > 60:       
        return f"{round(elapsed_time/60, ndigits=2)} minutes" 
    else:
        return f"{round(elapsed_time, ndigits=2)} seconds"

         