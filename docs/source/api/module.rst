.. _api.module:

lazyllm.Module
-----------------------

.. autoclass:: lazyllm.module.ModuleBase
    :members: forward, start, restart, update, evalset, eval, _get_train_tasks, _get_deploy_tasks
    :exclude-members:

.. autoclass:: lazyllm.module.ActionModule
    :members: evalset
    :exclude-members:

.. autoclass:: lazyllm.module.TrainableModule
    :members: start, restart, update, evalset, eval
    :exclude-members:

.. autoclass:: lazyllm.module.UrlModule
    :members: 
    :exclude-members:

.. autoclass:: lazyllm.module.ServerModule
    :members: start, restart, evalset
    :exclude-members:

.. autoclass:: lazyllm.module.TrialModule
    :members: start
    :exclude-members:

.. autoclass:: lazyllm.module.OnlineChatModule
    :members:
    :exclude-members:

.. autoclass:: lazyllm.module.OnlineEmbeddingModule
    :members:
    :exclude-members:

.. autoclass:: lazyllm.module.OnlineChatModuleBase
    :members:
    :exclude-members: forward

.. autoclass:: lazyllm.module.OnlineEmbeddingModuleBase
    :members:
    :exclude-members: forward
