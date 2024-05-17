from .rule import Rule, SearchMode

# KEYS
GPU_TYPE = Rule.from_indexed("GPU_TYPE", ["A100", "A800"])
GPU_NUM = Rule.from_indexed("GPU_NUM", [1, 2, 4, 8, 16, 24, 32, 64, 128, 256], matches=SearchMode.BINARY_FLOOR)
MODEL_NAME = Rule.from_indexed("MODEL_NAME",
    [
        "LLAMA_7B",
        "LLAMA_13B",
        "LLAMA_20B",
        "LLAMA_65B",
        "LLAMA2_7B",
        "LLAMA2_13B",
        "LLAMA2_70B",
        "INTERNLM2_7B",
        "INTERNLM2_20B",
        "STABLE_DIFFUSION_V1_5",
        "STABLE_DIFFUSION_V2",
    ],
)
CTX_LEN = Rule.from_indexed("CTX_LEN", [32, 64, 128, 256, 512, 1024, 2048, 4096], matches=SearchMode.BINARY_CEIL)
BATCH_SIZE = Rule.from_indexed("BATCH_SIZE", [1, 2, 4, 8, 16, 32, 64, 128], matches=SearchMode.BINARY_FLOOR)
TRAINABLE_PARAMS = Rule.from_indexed("TRAINABLE_PARAMS", [0, 1, 2, 5, 10, 15, 20, 25, 100], matches=SearchMode.BINARY_CEIL)

# VALUES
FRAMEWORK = Rule.from_options("FRAMEWORK", ["EASYLLM", "ALPACA", "COLLIE", "LIGHTLLM", "VLLM"])
TP = Rule.from_options("TP", [1, 2, 4, 8])
PP = Rule.from_options("PP", [1, 2, 4, 8])
ZERO = Rule.from_options("ZERO", [False, True])
GRADIENT_STEP = Rule.from_options("GRADIENT_STEP", [1, 2, 4, 8])
LORA_R = Rule.from_options("LORA_R", [0, 2, 4, 8, 16, 32])
MAX_TOKEN_NUM = Rule.from_options("MAX_TOKEN_NUM", [512, 1024, 2048, 4096, 8 * 1024, 12 * 1024, 16 * 1024, 24 * 1024])
MEMORY_USAGE_GB = Rule.from_type("MEMORY_USAGE_GB", int)
TGS = Rule.from_type("TGS", int)
ADDITIONAL_ARGUMENTS = Rule.from_type("ADDITIONAL_ARGUMENTS", str)

# RULE_SET
TRAINING_RULE_SET = [
    GPU_TYPE,
    GPU_NUM,
    MODEL_NAME,
    CTX_LEN,
    BATCH_SIZE,
    TRAINABLE_PARAMS,
    FRAMEWORK,
    TP,
    PP,
    ZERO,
    GRADIENT_STEP,
    LORA_R,
    MEMORY_USAGE_GB,
    TGS,
    ADDITIONAL_ARGUMENTS,
]

DEPLOY_RULE_SET = [
    GPU_TYPE,
    GPU_NUM,
    MODEL_NAME,
    CTX_LEN,
    BATCH_SIZE,
    TRAINABLE_PARAMS,
    FRAMEWORK,
    TP,
    MAX_TOKEN_NUM,
    TGS,
    ADDITIONAL_ARGUMENTS,
]
