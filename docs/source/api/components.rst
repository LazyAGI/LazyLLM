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

.. _api.components.prompter:

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

.. _api.components.register:

Register
=========

.. autofunction:: lazyllm.components.register

ModelManager
================

.. autoclass:: lazyllm.components.ModelManager
    :members: 
    :exclude-members:

Formatter
==========

.. autoclass:: lazyllm.components.formatter.LazyLLMFormatterBase
    :members:
    :exclude-members:

.. autoclass:: lazyllm.components.JsonFormatter
    :members:
    :exclude-members:

.. autoclass:: lazyllm.components.EmptyFormatter
    :members:
    :exclude-members:
