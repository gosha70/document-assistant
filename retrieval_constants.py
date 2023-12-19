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