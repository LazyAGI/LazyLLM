models_pool = {
    'chatglm3':{
        'lora_config':{
            'lora_target_modules': '[query_key_value,dense,dense_4h_to_h,dense_h_to_4h]',
            'modules_to_save': '[word_embeddings, output_layer]',
        },
        '6b': {
            'frame_propose':{
                1:['alpacalora', 'collie'],
                2:['alpacalora', 'collie'],
                4:['alpacalora', 'collie'],
                8:['alpacalora', 'collie'],
                16:['alpacalora'],
            },
            'gpu_memory': 11.18,
            'addition_parameters':{
                'micro_batch_size': 4,
                'batch_size': 32,
            }
        }
    },
    'internlm2':{
        'lora_config':{
            'lora_target_modules': '[wo,wqkv]',
            'modules_to_save': '[tok_embeddings,output]',
        },
        '7b': {
            'frame_propose':{
                1:['alpacalora', 'collie'],
                2:['alpacalora', 'collie'],
                4:['alpacalora', 'collie'],
                8:['alpacalora', 'collie'],
                16:['alpacalora'],
            },
            'gpu_memory': 13.04,
            'addition_parameters':{
                'micro_batch_size': 4,
                'batch_size': 32,
            }
        },
        '20b': {
            'frame_propose':{
                1:[],
                2:[],
                4:['alpacalora', 'collie'],
                8:['alpacalora', 'collie'],
                16:['alpacalora'],
            },
            'gpu_memory': 37.25,
            'addition_parameters':{
                'micro_batch_size': 2,
                'batch_size': 32,
            }
        }
    },
}