::: lazyllm.flow.FlowBase
    members: is_root, ancestor, for_each, id
    exclude-members:

::: lazyllm.flow.LazyLLMFlowsBase
    members: 
    - register_hook
    - unregister_hook
    - clear_hooks
    - set_sync
    - wait
    - invoke
    - bind
    exclude-members:

::: lazyllm.flow.Pipeline
    members: output
    exclude-members:

::: lazyllm.flow.save_pipeline_result

::: lazyllm.flow.Parallel
    members: join, sequential
    exclude-members:

::: lazyllm.flow.Diverter
    members: 
    exclude-members:

::: lazyllm.flow.Warp
    members: 
    exclude-members:

::: lazyllm.flow.IFS
    members: 
    exclude-members:

::: lazyllm.flow.Switch
    members: 
    exclude-members:

::: lazyllm.flow.Loop
    members: 
    exclude-members:

::: lazyllm.flow.Graph
    members: Node, set_node_arg_name, start_node, end_node, add_edge, add_const_edge, topological_sort, compute_node
    exclude-members: