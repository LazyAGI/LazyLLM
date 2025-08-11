
::: lazyllm.module.ModuleBase
    options:
      members:
      - _get_deploy_tasks
      - _get_train_tasks
      - stream_output
      - used_by
      - forward
      - register_hook
      - unregister_hook
      - clear_hooks
      - start
      - restart
      - update
      - update_server
      - evalset
      - eval
      - wait
      - stop
      - for_each
        
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

::: lazyllm.module.AutoModel
    options:
      members:

::: lazyllm.module.TrialModule
    members: start
    exclude-members:

::: lazyllm.module.OnlineChatModule
    members:
    exclude-members:

::: lazyllm.module.OnlineEmbeddingModule
    members:
    exclude-members:

::: lazyllm.module.OnlineChatModuleBase
    options:
      members:
      filters:
      - '!forward'

::: lazyllm.module.OnlineEmbeddingModuleBase
    members:
    exclude-members: forward

