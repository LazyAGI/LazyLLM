models_config = {
    'chatglm3': {
        'lora_target_modules': '[query_key_value,dense,dense_4h_to_h,dense_h_to_4h]',
        'modules_to_save': '[word_embeddings, output_layer]',
    },
    'internlm2': {
        'lora_target_modules': '[wo,wqkv]',
        'modules_to_save': '[tok_embeddings,output]',
    },
}
