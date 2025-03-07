from .rule import Rule, SearchMode

# KEYS
GPU_TYPE = Rule.from_indexed("GPU_TYPE", ["H100", "A100", "A800"])
GPU_NUM = Rule.from_indexed("GPU_NUM", [1, 2, 4, 8], matches=SearchMode.BINARY_FLOOR)
MODEL_NAME = Rule.from_indexed("MODEL_NAME", ["LLAMA_7B", "LLAMA_20B", "LLAMA_70B", "LLAMA_100B"])
CTX_LEN = Rule.from_indexed("CTX_LEN", [512, 2048], matches=SearchMode.BINARY_CEIL)
BATCH_SIZE = Rule.from_indexed("BATCH_SIZE", [1, 8, 32, 128, 512], matches=SearchMode.BINARY_CEIL)
MAX_TOKEN_NUM = Rule.from_indexed("MAX_TOKEN_NUM", [64 * 1024, 256 * 1024, 1024 * 1024],
                                  matches=SearchMode.BINARY_CEIL)
LORA_R = Rule.from_indexed("LORA_R", [8, 16, 32], matches=SearchMode.BINARY_CEIL)

# VALUES
FINETUNE_FRAMEWORK = Rule.from_options("FRAMEWORK", ["ALPACALORA", "COLLIE", 'LLAMAFACTORY'])
DEPLOY_FRAMEWORK = Rule.from_options("FRAMEWORK", ["LIGHTLLM", "VLLM"])
TP = Rule.from_options("TP", [1, 2, 4, 8])
ZERO = Rule.from_options("ZERO", [0, 1, 2, 3])
GRADIENT_STEP = Rule.from_options("GRADIENT_STEP", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
TGS = Rule.from_type("TGS", int)

# RULE_SET
FINETUNE_RULE_SET = [
    # Keys
    GPU_TYPE, GPU_NUM, MODEL_NAME, CTX_LEN, BATCH_SIZE, LORA_R,
    # Values
    FINETUNE_FRAMEWORK, TP, ZERO, GRADIENT_STEP, TGS,
]

DEPLOY_RULE_SET = [
    # Keys
    GPU_TYPE, GPU_NUM, MODEL_NAME, MAX_TOKEN_NUM,
    # Values
    DEPLOY_FRAMEWORK, TP, TGS,
]
