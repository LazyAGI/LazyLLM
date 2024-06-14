lazyllm.Components
-----------------------

Finetune
=========

.. automodule:: lazyllm.components.finetune
    :members: 
    :exclude-members:

.. automodule:: lazyllm.components.auto
    :members: AutoFinetune
    :exclude-members:

Deploy
=========

.. automodule:: lazyllm.components.deploy
    :members: 
    :exclude-members:

.. automodule:: lazyllm.components.auto
    :members: AutoDeploy
    :exclude-members:

Launcher
=========

.. automodule:: lazyllm.launcher
    :members: 
    :exclude-members: Status, get_idle_nodes

Prompter
=========

.. autoclass:: lazyllm.components.prompter.LazyLLMPrompterBase
    :members: generate_prompt, get_response
    :exclude-members:

.. autoclass:: lazyllm.components.AlpacaPrompter
    :members: generate_prompt, get_response
    :exclude-members:

.. autoclass:: lazyllm.components.ChatPrompter
    :members: generate_prompt, get_response
    :exclude-members:

Register
=========

.. autofunction:: lazyllm.components.register

ModelDownloader
================

.. autoclass:: lazyllm.components.ModelDownloader
    :members: 
    :exclude-members:
