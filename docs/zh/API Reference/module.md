::: lazyllm.module.ModuleBase
    options:
      members:
      - eval
      - evalset
      - forward
      - start
      - restart
      - update
      - stream_output
      - used_by
      - register_hook
      - unregister_hook
      - clear_hooks
      - update_server
      - wait
      - stop
      - for_each

::: lazyllm.module.servermodule.LLMBase
    options:
      members:
      - prompt
      - formatter
      - share

::: lazyllm.module.ActionModule
    options:
      members:
      - evalset
      - forward
      - submodules

::: lazyllm.module.TrainableModule
    options:
      members:
      - start
      - restart
      - update
      - evalset
      - eval
      - wait
      - stop
      - prompt
      - log_path
      - forward_openai
      - forward_standard
      - forward

::: lazyllm.module.UrlModule
    options:
      members:
      - forward

::: lazyllm.module.ServerModule
    options:
      members:
      - start
      - restart
      - evalset
      - wait
      - stop

::: lazyllm.module.AutoModel
    options:
      members:

::: lazyllm.module.TrialModule
    members: [start]
    exclude-members:

::: lazyllm.module.OnlineChatModule
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoChat
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.ppio.PPIOChat
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoMultiModal
    members:
    exclude-members:

::: lazyllm.module.OnlineEmbeddingModule
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.openai.OpenAIEmbed
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenSTT
    members:
    exclude-members:

::: lazyllm.module.OnlineChatModuleBase
    options:
      members:
      - set_train_tasks
      - set_specific_finetuned_model

::: lazyllm.module.OnlineEmbeddingModuleBase
    members:
        - forward
        - run_embed_batch

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoEmbed
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoMultimodal_Embed
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMChat
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMText2Image
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenText2Image
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.kimi.KimiChat
    options:
      members:

::: lazyllm.module.llms.onlinemodule.fileHandler.FileHandlerBase
    members: get_finetune_data
    exclude-members: 

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMChat
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMRerank
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMMultiModal
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenRerank
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenTTS
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.sensenova.SenseNovaChat
    members: set_deploy_parameters
    exclude-members:

::: lazyllm.module.llms.onlinemodule.base.onlineMultiModalBase.OnlineMultiModalBase
    members:
    exclude-members:
    options:
      members:
        - get_finetune_data 

::: lazyllm.module.llms.onlinemodule.base.utils.LazyLLMOnlineBase
    members:
    exclude-members:

::: lazyllm.module.module.ModuleCache
    members: [get, set, close]
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenChat
    options:
      members:
        - set_deploy_parameters

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenEmbed
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMEmbed
    options:
      members: 

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMSTT
    options:
      members: 

::: lazyllm.module.llms.onlinemodule.supplier.deepseek.DeepSeekChat
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoText2Image
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.openai.OpenAIChat
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.openai.OpenAIRerank
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.sensenova.SenseNovaEmbed
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.siliconflow.SiliconFlowTTS
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.siliconflow.SiliconFlowChat
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.siliconflow.SiliconFlowRerank
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.siliconflow.SiliconFlowText2Image
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.aiping.AipingChat
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.aiping.AipingEmbed
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.aiping.AipingRerank
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.aiping.AipingText2Image
    options:
      members:
