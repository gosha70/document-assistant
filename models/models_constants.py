
import torch

DEVICE_MAP = "auto"

N_CTX = 6144

# How many tokens are processed in parallel, default is 8, set to bigger number.
# Should be between 1 and N_CTX, consider the amount of VRAM in your GPU.
N_BATCH = 512 

# Determines how many layers of the model are offloaded to your Metal GPU, in the most case, 
# set it to 1 is enough for Metal.
# Change this value based on your model and your GPU VRAM pool.
N_GPU_LAYERS = 40 

# Embedding settings
EMBEDDING_KWARGS = {'device': 'cpu'}
ENCODE_KWARG = {'normalize_embeddings': True}

DEFAULT_MODEL_NAME = "hkunlp/instructor-large" 
DEFAULT_MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
DEFAULT_MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

MISTRAL_SMALL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MISTRAL_SMALL_MODEL_BASENAME = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
MISTRAL_LARGE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MISTRAL_LARGE_MODEL_BASENAME = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"

# Device Type
DEVICE_TYPE_MPS = "mps"
DEVICE_TYPE_CUDA = "cuda"
DEVICE_TYPE_CPU = "cpu"

if torch.backends.mps.is_available():
    DEVICE_TYPE = DEVICE_TYPE_MPS
elif torch.cuda.is_available():
    DEVICE_TYPE = DEVICE_TYPE_CUDA
else:
    DEVICE_TYPE = DEVICE_TYPE_CPU