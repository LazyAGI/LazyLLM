### Building Complex Applications

### Building Applications Based on Internal Tools

!!! Note

    Although it appears to be composed of modules combined into a data flow and then started uniformly, we are not a "static graph". The main manifestations are:

    - We use basic Python syntax to build applications, and the usage is very close to traditional programming methods.
    - You can still take advantage of Python's powerful and flexible features to change the topology of an already built application during runtime, and subsequent execution will follow the modified structure.
    - You can flexibly inject hook functions you wish to execute at the connection points of the modules. These functions can even be defined at runtime.

Here is an example of an application built based on LazyLLM:

- [Chatbot](../Cookbook/robot.md)
    - Module：[TrainableModule][lazyllm.module.TrainableModule]
    - Tools：[WebModule][lazyllm.tools.WebModule]
    - Flow：None
- [Painting Master](../Cookbook/painting_master.md)
    - Module：[TrainableModule][lazyllm.module.TrainableModule]
    - Tools：[WebModule][lazyllm.tools.WebModule]
    - Flow：[Pipeline][lazyllm.flow.Pipeline]
- [Multimodal Robot](../Cookbook/multimodal_robot.md)
    - Module：[TrainableModule][lazyllm.module.TrainableModule]
    - Tools：[WebModule][lazyllm.tools.WebModule]
    - Flow：[Pipeline][lazyllm.flow.Pipeline], [Switch][lazyllm.flow.Switch]
- [Great Writer](../Cookbook/great_writer.md)
    - Module：[TrainableModule][lazyllm.module.TrainableModule]
    - Tools：[WebModule][lazyllm.tools.WebModule]
    - Flow：[Pipeline][lazyllm.flow.Pipeline], [Warp][lazyllm.flow.Warp]
- [Knowledge Base Q&A Assistant](../Cookbook/rag.md)
    - Module：[TrainableModule][lazyllm.module.TrainableModule]
    - Tools：[WebModule][lazyllm.tools.WebModule], [Document][lazyllm.tools.Document], [Retriever][lazyllm.tools.Retriever], [Reranker][lazyllm.tools.Reranker]
    - Flow：[Pipeline][lazyllm.flow.Pipeline], [Parallel][lazyllm.flow.Parallel]

### Building Applications with Open Source Tools

### Typical Scenario Applications
