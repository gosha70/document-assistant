# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Modelfile  extensions: 
GGUF_EXTENSION = ".gguf"
GGML_EXTENSION = ".ggml"
AWQ_EXTENSION = ".awq"

# Chain types:
CHAIN_TYPE_STUFF="stuff"
CHAIN_TYPE_MAP_REDUCE="map_reduce"
CHAIN_TYPE_MAP_RERANK="map_rerank"

CACHE_DIR="./model_cache/"